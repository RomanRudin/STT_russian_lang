import os
import json
import logging
import random
from typing import Dict, Any, List
from datasets import Dataset, DatasetDict, Audio

logger = logging.getLogger(__name__)

def _read_jsonl_manifest(manifest_path: str, root_dir: str, max_dur: float, min_dur: float, allowed_subsets: List[str] = None) -> List[Dict[str, str]]:
    """Читает манифест, фильтрует по сабсетам и ищет аудиофайлы."""
    records = []
    manifest_dir = os.path.dirname(manifest_path)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            rel_path = data.get("audio_filepath", "")
            text = data.get("text", "").strip()
            duration = data.get("duration", 0.0)
            
            # 1. Фильтрация по длительности
            if not (min_dur <= duration <= max_dur):
                continue
            if not text:
                continue
                
            # 2. Фильтрация по сабсетам (crowd / farfield)
            # В train один общий манифест, поэтому проверяем, есть ли слово "crowd" в пути
            if allowed_subsets:
                if not any(sub in rel_path or sub in manifest_path for sub in allowed_subsets):
                    continue
                    
            # 3. Умный поиск аудиофайла
            candidates = [
                os.path.join(manifest_dir, rel_path),                          # Относительно самого манифеста
                os.path.join(root_dir, rel_path),                              # Относительно корня
                os.path.join(root_dir, "train", rel_path),                     # Если путь от train
                os.path.join(root_dir, "test", rel_path),                      # Если путь от test
                os.path.join(manifest_dir, "files", os.path.basename(rel_path)) # Для папки test/crowd/files/
            ]
            
            abs_path = None
            for c in candidates:
                if os.path.isfile(c):
                    abs_path = c
                    break
                # Fallback: если манифест просит .wav, а у нас .opus
                c_opus = c.replace('.wav', '.opus')
                if os.path.isfile(c_opus):
                    abs_path = c_opus
                    break
                    
            if abs_path:
                records.append({"audio": abs_path, "sentence": text})
                
    return records

def create_golos(cfg: Dict[str, Any]) -> DatasetDict:
    """Собирает датасет Голос под специфичную структуру train/test."""
    golos_cfg = cfg['dataset']['sources']['golos']
    root_path = golos_cfg.get('path')
    
    if not root_path or not os.path.isdir(root_path):
        raise FileNotFoundError(f"Корневая папка с Голосом не найдена: {root_path}")

    subsets = golos_cfg.get('subsets', ['crowd'])
    max_dur = cfg['dataset']['params']['max_duration']
    min_dur = cfg['dataset']['params']['min_duration']
    sample_rate = cfg['dataset']['params']['sample_rate']

    logger.info("Загрузка датасета Голос из %s, сабсеты: %s", root_path, subsets)

    train_records = []
    test_records = []

    # --- 1. ЗАГРУЗКА TRAIN ---
    # В train лежит общий manifest.json(l). Ищем его.
    train_manifest = os.path.join(root_path, "train", "manifest.jsonl")
    if not os.path.isfile(train_manifest):
        train_manifest = os.path.join(root_path, "train", "manifest.json") # Ты писал, что у тебя .json
        
    if os.path.isfile(train_manifest):
        logger.info(f"Читаем train манифест: {train_manifest}")
        train_records.extend(_read_jsonl_manifest(train_manifest, root_path, max_dur, min_dur, subsets))
    else:
        logger.warning(f"Манифест train не найден по пути: {train_manifest}")

    # --- 2. ЗАГРУЗКА TEST (используем как Validation) ---
    # В test лежат отдельные папки crowd и farfield с их манифестами
    for subset in subsets:
        test_manifest = os.path.join(root_path, "test", subset, "manifest.jsonl")
        if not os.path.isfile(test_manifest):
            test_manifest = os.path.join(root_path, "test", subset, "manifest.json")
            
        if os.path.isfile(test_manifest):
            logger.info(f"Читаем test манифест: {test_manifest}")
            test_records.extend(_read_jsonl_manifest(test_manifest, root_path, max_dur, min_dur, [subset]))

    if not train_records:
        raise RuntimeError(f"Не найдено тренировочных аудиозаписей в {root_path}. Проверь структуру папок и пути.")

    # Перемешиваем тренировочную выборку
    random.shuffle(train_records)

    max_train = cfg['dataset'].get('max_train_samples')
    max_eval = cfg['dataset'].get('max_eval_samples')

    if max_train and max_train < len(train_records):
        train_records = train_records[:max_train]
        logger.info("Тренировочная выборка урезана до %d примеров", max_train)

    if test_records and max_eval and max_eval < len(test_records):
        test_records = test_records[:max_eval]
        logger.info("Тестовая выборка урезана до %d примеров", max_eval)

    def to_dataset(records):
        ds = Dataset.from_list(records)
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        return ds

    result_dict = {"train": to_dataset(train_records)}
    
    if test_records:
        result_dict["validation"] = to_dataset(test_records)
    else:
        val_frac = golos_cfg.get('val_fraction', 0.02)
        n_val = max(1, int(len(train_records) * val_frac))
        result_dict["validation"] = to_dataset(train_records[:n_val])
        result_dict["train"] = to_dataset(train_records[n_val:])

    result = DatasetDict(result_dict)
    logger.info("Голос загружен. Сплиты: %s", {k: len(v) for k, v in result.items()})
    return result
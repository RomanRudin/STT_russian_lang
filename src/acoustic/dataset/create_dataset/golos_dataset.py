import os
import json
import tarfile
from datasets import Dataset, DatasetDict, Audio

def create_golos(cfg):
    golos_cfg = cfg['dataset']['sources']['golos']
    raw_tar_path = golos_cfg.get('tar_path', 'golos_opus.tar')
    
    tar_path = raw_tar_path
    if not os.path.exists(tar_path):
        alt_path = os.path.join("..", raw_tar_path)
        if os.path.exists(alt_path):
            tar_path = alt_path
        else:
            raise FileNotFoundError(f"Архив не найден: '{raw_tar_path}'")
            
    val_frac = golos_cfg.get('val_fraction', 0.02)
    params = cfg['dataset'].get('params', {})
    max_dur = params.get('max_duration', 30.0)
    min_dur = params.get('min_duration', 0.5)
    sample_rate = params.get('sample_rate', 16000)
    
    max_samples = cfg['dataset'].get('max_train_samples')
    
    print("Сканирую манифесты внутри архива...")
    manifest_data = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if member.name.endswith(".jsonl") or member.name.endswith(".json"):
                f = tar.extractfile(member)
                if f is not None:
                    for line in f:
                        decoded_line = line.decode('utf-8')
                        if not decoded_line.strip():
                            continue
                        data = json.loads(decoded_line)
                        rel_path = data.get("audio_filepath", "")
                        text = data.get("text", "").strip()
                        duration = data.get("duration", 0.0)
                        
                        if min_dur <= duration <= max_dur and text:
                            file_name = os.path.basename(rel_path)
                            manifest_data[file_name] = text
                            
                            # ЖЕСТКИЙ ТОРМОЗ: если набрали лимит, выходим из цикла!
                            if max_samples and len(manifest_data) >= max_samples:
                                break
            
            # Выходим из внешнего цикла архива, если лимит уже достигнут
            if max_samples and len(manifest_data) >= max_samples:
                break
                            
    print(f"Найдено {len(manifest_data)} текстов. Начинаю потоковую сборку...")

    def gen():
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isreg() or not member.name.endswith('.opus'):
                    continue
                
                file_name = os.path.basename(member.name)
                if file_name in manifest_data:
                    audio_bytes = tar.extractfile(member).read()
                    
                    yield {
                        "audio": {"path": member.name, "bytes": audio_bytes},
                        "sentence": manifest_data[file_name]
                    }
                    
    ds = Dataset.from_generator(gen)
    
    # Отдаем сырые байты
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate, decode=False))
    
    ds = ds.shuffle(seed=42)
    split_ds = ds.train_test_split(test_size=val_frac, seed=42)
    
    result = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"]
    })
    
    return result
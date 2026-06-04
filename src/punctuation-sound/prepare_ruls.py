"""
prepare_ruls.py
===============
Автономная подготовка датасета Russian LibriSpeech (RuLS) БЕЗ HuggingFace Hub.

Зачем: HF Hub может быть недоступен из вашей сети. Этот скрипт качает
ОФИЦИАЛЬНЫЙ архив RuLS с OpenSLR (зеркала US/EU/CN — CN обычно открывается
там, где HF заблокирован), распаковывает, парсит транскрипции и сохраняет
готовые Example в один .pkl. Дальше ноутбук просто их загружает.

КЛЮЧЕВОЕ: берём поле text_no_preprocessing (оригинал С пунктуацией), а не
нормализованный text (без пунктуации) — иначе восстанавливать будет нечего.

Запуск (из папки проекта, тем же python, что у Jupyter-ядра):
    python prepare_ruls.py                 # скачать + подготовить train+val
    python prepare_ruls.py --limit 8000    # ограничить число клипов train
    python prepare_ruls.py --no-download    # архив уже скачан вручную в cache_dir

Если автозагрузка не идёт, скачайте вручную (зеркало CN):
    https://openslr.magicdatatech.com/resources/96/ruls_data.tar.gz
положите в ./.cache/ruls_data.tar.gz и запустите с --no-download.

Результат: mailabs_examples.pkl  -> по умолчанию имя ruls_examples.pkl.
"""

import os
import sys
import json
import pickle
import tarfile
import argparse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.data import parse_transcription, Example   # noqa: E402
from modules.config import ACOUSTIC_DIM, DataConfig      # noqa: E402
import numpy as np                                       # noqa: E402

MIRRORS = [
    "https://www.openslr.org/resources/96/ruls_data.tar.gz",
    "https://openslr.elda.org/resources/96/ruls_data.tar.gz",
    "https://openslr.magicdatatech.com/resources/96/ruls_data.tar.gz",  # CN
]
# приоритет текстовых полей: сначала с пунктуацией
TEXT_FIELDS = ["text_no_preprocessing", "transcript", "sentence", "text"]


def download(dst):
    if os.path.exists(dst):
        print(f"[skip] архив уже скачан: {dst}")
        return True
    for url in MIRRORS:
        try:
            print(f"[download] {url}  (~9 ГБ, разовая операция)")
            def hook(b, bs, t):
                if t > 0:
                    sys.stdout.write(f"\r  {min(100, b*bs*100//t):3d}%  ({b*bs/1e6:7.1f} МБ)")
                    sys.stdout.flush()
            urllib.request.urlretrieve(url, dst, reporthook=hook)
            print("\n[ok] скачано.")
            return True
        except Exception as e:
            print(f"\n  зеркало не сработало: {e}")
    print("[error] не удалось скачать ни с одного зеркала.")
    print("  Скачайте вручную (CN-зеркало) и используйте --no-download.")
    return False


def extract(tgz, out_dir):
    marker = os.path.join(out_dir, "_extracted.ok")
    if os.path.exists(marker):
        print(f"[skip] уже распакован: {out_dir}")
        return True
    try:
        print("[extract] распаковка (несколько минут) ...")
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(out_dir)
        open(marker, "w").close()
        print("[ok] распаковано.")
        return True
    except Exception as e:
        print(f"[error] распаковка не удалась: {e}")
        return False


def pick_text(row):
    for fld in TEXT_FIELDS:
        v = row.get(fld)
        if isinstance(v, str) and v.strip():
            return v, fld
    return None, None


def iter_manifests(root):
    """Идёт по JSON/JSONL-манифестам RuLS и выдаёт (row, audio_path, split)."""
    def split_of(name):
        n = name.lower()
        if "test" in n: return "test"
        if "dev" in n or "val" in n: return "validation"
        return "train"
    for cur, _d, files in os.walk(root):
        for fn in files:
            if not (fn.endswith(".json") or fn.endswith(".jsonl")):
                continue
            split = split_of(fn)
            try:
                with open(os.path.join(cur, fn), encoding="utf-8") as f:
                    content = f.read().strip()
                rows = []
                try:
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                except Exception:
                    obj = json.loads(content)
                    rows = obj if isinstance(obj, list) else [obj]
                for r in rows:
                    if isinstance(r, dict):
                        af = r.get("audio_filepath") or r.get("audio") or r.get("wav") or ""
                        wav = af if os.path.isabs(af) else os.path.join(cur, af)
                        yield r, wav, split
            except Exception:
                continue


def build(root, limit_train=None):
    counts = {"train": 0, "validation": 0, "test": 0}
    used_field = None
    by_split = {"train": [], "validation": [], "test": []}
    for row, wav, split in iter_manifests(root):
        text, fld = pick_text(row)
        if not text:
            continue
        used_field = used_field or fld
        if limit_train and split == "train" and counts["train"] >= limit_train:
            continue
        words, punct, para, cap = parse_transcription(text)
        if len(words) < 3:
            continue
        ex = Example(words=words, punct_ids=punct, para_ids=para, cap_ids=cap,
                     acoustic=np.zeros((len(words), ACOUSTIC_DIM), np.float32),
                     has_acoustic=False, meta={"audio_path": wav, "split": split})
        by_split[split].append(ex)
        counts[split] += 1
    print(f"[build] использовано текстовое поле: {used_field}")
    print(f"[build] train={counts['train']} val={counts['validation']} test={counts['test']}")
    if used_field and used_field != "text_no_preprocessing":
        print("  !! ВНИМАНИЕ: поле text_no_preprocessing не найдено, взято", used_field)
        print("     Если это нормализованный текст без пунктуации — обучать пунктуацию нельзя.")
    return by_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="макс. клипов train")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--cache", default="./.cache")
    ap.add_argument("--out", default="ruls_examples.pkl")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    tgz = os.path.join(args.cache, "ruls_data.tar.gz")
    out_dir = os.path.join(args.cache, "ruls_data")

    if not args.no_download:
        if not download(tgz):
            sys.exit(1)
    elif not os.path.exists(tgz):
        print(f"[error] {tgz} не найден. Уберите --no-download или скачайте архив.")
        sys.exit(1)

    if not extract(tgz, out_dir):
        sys.exit(1)

    by_split = build(out_dir, limit_train=args.limit)
    total = sum(len(v) for v in by_split.values())
    if total == 0:
        print("[error] не найдено примеров. Проверьте структуру архива.")
        sys.exit(1)

    with open(args.out, "wb") as f:
        pickle.dump(by_split, f)
    print(f"[done] сохранено: {args.out}")
    print("В ноутбуке:")
    print("  import pickle")
    print(f"  d = pickle.load(open('{args.out}','rb'))")
    print("  train_examples, val_examples = d['train'], d['validation']")


if __name__ == "__main__":
    main()

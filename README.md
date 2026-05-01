# STT_russian_lang
Speach-to-text for Russian language


### Setup

```bash
python -m venv venv 
venv/bin/activate
pip install -r requirements.txt
```

## Downloading datasets

https://cdn.chatwm.opensmodel.sberdevices.ru/golos/golos_opus.tar



### Running acoustic fine-tuning

```bash
python scripts/preprocess.py 

python -m src.acoustic.whisper.train --lora
```

### Comnparison of different models

```bash
python scripts/compare_experiments.py

# Sort by a different metric
python scripts/compare_experiments.py --sort cer

# Diff two experiments (config differences and metric side-by-side)
python scripts/compare_experiments.py --diff large_full medium_lora

# Export to CSV
python scripts/compare_experiments.py --csv results.csv
```




https://huggingface.co/openai/whisper-large-v3-turbo
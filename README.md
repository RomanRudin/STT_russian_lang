# STT_russian_lang
Speach-to-text for Russian language


## WORKING IN REPO

Please, make commit messages READABLE, UNDERSTANDABLE, SELF-EXPLANATORY and on English.

```bash
# Starting
git clone https://github.com/RomanRudin/STT_russian_lang
git branch NAME_OF_YPUR_BRANCH
git checkout NAME_OF_YPUR_BRANCH

# Pulling
git pull

# I think I'd be doing merging but for the sake of fullness
git merge NAME_OF_YPUR_BRANCH
```

# Acoustic model
### Setup

```bash
python -m venv venv 
venv/bin/activate
pip install -r requirements.txt
```

<!-- ## Downloading datasets

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




https://huggingface.co/openai/whisper-large-v3-turbo -->
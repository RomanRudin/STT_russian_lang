"""
Augmentation: add Gaussian noise and/or random speed perturbation.
Uses parameters passed directly (not from global cfg).
"""

import random
import torch
import torchaudio
from datasets import DatasetDict
from typing import Dict, Any
from tqdm import tqdm


def apply(dataset: DatasetDict, params: Dict[str, Any]) -> DatasetDict:
    if not params.get('enabled', True):
        return dataset

    noise_prob = params.get('noise_prob', 0.3)
    noise_min = params.get('noise_level_min', 0.001)
    noise_max = params.get('noise_level_max', 0.015)
    speed_prob = params.get('speed_prob', 0.2)
    speed_range = params.get('speed_range', [0.9, 1.1])

    def augment_example(example):
        wav = torch.tensor(example['audio']['array']).unsqueeze(0)  # (1, T)
        if random.random() < noise_prob:
            noise = torch.randn_like(wav) * random.uniform(noise_min, noise_max)
            wav = wav + noise
        if random.random() < speed_prob:
            factor = random.uniform(*speed_range)
            orig_freq = example['audio']['sampling_rate']
            temp_freq = int(orig_freq * factor)
            wav = torchaudio.functional.resample(wav, orig_freq, temp_freq)
            wav = torchaudio.functional.resample(wav, temp_freq, orig_freq)
        example['audio']['array'] = wav.squeeze(0).numpy()
        return example

    augmented = DatasetDict()
    for split, ds in tqdm(dataset.items(), desc="Applying augmentation"):
        augmented[split] = ds.map(augment_example)
    return augmented
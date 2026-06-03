import sys
from pathlib import Path
import torch

script_dir = Path(__file__).resolve().parent  # src/
checkpoints_root = script_dir / "acoustic" / "checkpoints"

final_dir = checkpoints_root / "final_model"
best_dir = checkpoints_root / "checkpoint-160"

def load_state_dict_from_checkpoint(ckpt_dir: Path):
    """Загружает state_dict из папки чекпоинта."""
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Directory {ckpt_dir} does not exist.\n"
            f"Contents of {ckpt_dir.parent}: {list(ckpt_dir.parent.iterdir()) if ckpt_dir.parent.exists() else 'parent not found'}"
        )
    # Сначала safetensors, потом pytorch_model.bin, потом sharded
    safetensor = ckpt_dir / "model.safetensors"
    if safetensor.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensor))
    bin_file = ckpt_dir / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(str(bin_file), map_location="cpu")
    shards = sorted(ckpt_dir.glob("pytorch_model-*.bin"))
    if shards:
        state_dict = {}
        for shard in shards:
            state_dict.update(torch.load(str(shard), map_location="cpu"))
        return state_dict
    raise FileNotFoundError(f"No model weights found in {ckpt_dir}. Files: {list(ckpt_dir.iterdir())}")

try:
    state_final = load_state_dict_from_checkpoint(final_dir)
    state_best = load_state_dict_from_checkpoint(best_dir)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

all_match = True
for key in state_final.keys():
    if not torch.allclose(state_final[key], state_best[key], atol=1e-7):
        print(f"Mismatch in {key}")
        all_match = False

if all_match:
    print("Weights are identical - final_model contains the best checkpoint.")
else:
    print("Weights differ - final_model does not match the best checkpoint.")
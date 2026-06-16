#pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm5.7
#pip install transformers==4.37.2 accelerate==0.28.0 tokenizers==0.15.2 peft==0.10.0 datasets==2.18.0 evaluate==0.4.6 numpy==1.26.0 librosa==0.10.2 soundfile==0.12.1 jiwer==3.0.3 pyyaml==6.0.3 tensorboard==2.17.0 matplotlib==3.7.2 jupyter==1.0.0 ipywidgets==8.1.5
#pip install sentencepiece==0.2.0
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

try:
    x = torch.tensor([1, 2, 3], device="cuda")
    print("Tensor created on GPU, no error.")
except Exception as e:
    print(f"Error creating tensor: {e}")
    exit()

try:
    embedding = torch.nn.Embedding(1000, 128).to("cuda")
    input_ids = torch.randint(0, 1000, (4, 16), device="cuda")
    out = embedding(input_ids)
    print("Embedding forward pass succeeded on GPU.")
except Exception as e:
    print(f"Embedding forward pass FAILED: {e}")
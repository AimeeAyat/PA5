"""Quick CUDA availability check"""
import torch

print("="*60)
print("CUDA AVAILABILITY CHECK")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("\nCUDA is NOT available. Possible reasons:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. NVIDIA drivers not installed")
    print("3. CUDA toolkit not installed or version mismatch")
    print("\nTo fix:")
    print("  - Install CUDA-enabled PyTorch:")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  - Or check NVIDIA drivers:")
    print("    nvidia-smi")

print("="*60)

import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.device_count())  # Should be > 0
print(torch.cuda.get_device_name(0))  # Name of the GPU

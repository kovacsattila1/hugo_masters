import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Create a tensor and move it to the GPU
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
print(f"Tensor on device: {tensor.device}")

# Perform a basic operation
result = tensor * 2
print(f"Result tensor: {result}")
print(f"Result tensor on device: {result.device}")

# Verify if the tensor is on the GPU
if result.is_cuda:
    print("The result tensor is on the GPU.")
else:
    print("The result tensor is on the CPU.")

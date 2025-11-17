## TEST: If GPU is available
import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Num GPUs Available: {num_gpus}")

    for i in range(num_gpus):
        print(f" --- GPU {i}: {torch.cuda.get_device_name(i)}")

else:
    print("CUDA is not available. Running on CPU...")
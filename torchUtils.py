import torch


def getDevice() -> torch.device:
    """
    Returns the device to use for torch: cuda, mps or cpu
    Note for mps: as of today you need to set PYTORCH_ENABLE_MPS_FALLBACK=1 in os env
      with conda:
        conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
    """
    if torch.cuda.is_available():
        print("Using CUDA with PyTorch")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS with PyTorch")
        return torch.device("mps")
    else:
        print("No CUDA or MPS acceleration available for PyTorch. Using CPU.")
        return torch.device("cpu")

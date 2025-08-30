"""
Verify the MPS with PyTorch
"""

import torch     # type: ignore
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS backend is not available. Please check your PyTorch installation.")
    mps_device = None
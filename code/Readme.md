## Accelerated PyTorch training on Mac

### Metal Accelaration

PyTorch uses the Metal perfromance shaders (MPS) backend for GPU training accelaration. This MPS backend extends the PyTorch framework, providing frameworks with kernels that are fine-tuned for the unique characteristics of each Metal GPU family.


### Installing

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```


### Verify

```python
import torch
if torch.backends.mps.is_available():
    mps_time = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)

else:
    print("MPS device not found")
```


The output should show:

```bash
tensor([1.], device ='mps:0')
```
import os

os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
import torch

# Base lib for torch xla.
import torch_xla

# Core lib for torch xla.
import torch_xla.core.xla_model as xm

# Gets a xla device.
dev = xm.xla_device()

# Execute randn to TPU
t = torch.randn(2, 2, device=dev)
print(t.device)
print(t)

# Execute randn to TPU
t0 = torch.randn(2, 2, device=dev)
# Execute randn to TPU
t1 = torch.randn(2, 2, device=dev)
print(t0 + t1)

# Execute randn to TPU
l_in = torch.randn(10, device=dev)
linear = torch.nn.Linear(10, 20)
l_out = linear(l_in)
print(l_out)

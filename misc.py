import torch

t = torch.randn(2, 2)
print(t.device)
print(t)

t0 = torch.randn(2, 2)
t1 = torch.randn(2, 2)
print(t0 + t1)

l_in = torch.randn(10)
linear = torch.nn.Linear(10, 20)
l_out = linear(l_in)
print(l_out)
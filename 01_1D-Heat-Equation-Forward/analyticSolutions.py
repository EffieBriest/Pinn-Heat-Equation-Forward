import torch

def  u(alpha,x,t):
    return torch.exp(-alpha*torch.pi**2*t)*torch.sin(torch.pi*x)


import torch
num_timesteps = 100
b = 64

t = torch.randint(0, num_timesteps, (b,)).long()

print(t)
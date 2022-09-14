import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

from adversarial_diffusion.adv_diffusion import AdvDiffusion

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

adv_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

image_sample = torch.rand([1, 3, 128, 128]).cuda()

print(image_sample)

print(model(image_sample, torch.tensor([0]).cuda()))

print(adv_model(image_sample, torch.tensor([0]).cuda()))


diffusion = AdvDiffusion(
    model,
    adv_model,
    image_size=128,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference)
    loss_type="l1",  # L1 or L2
).cuda()

loss = diffusion(image_sample)

print(loss)


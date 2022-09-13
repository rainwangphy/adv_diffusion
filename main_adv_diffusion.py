import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

adv_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

image_sample = torch.rand([1, 3, 128, 128]).cuda()

print(image_sample)

print(model(image_sample, torch.tensor([0]).cuda()))

print(adv_model(image_sample, torch.tensor([0]).cuda()))
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
from denoising_diffusion_pytorch import Unet
from adversarial_diffusion.trainer import Trainer

from adversarial_diffusion.adv_diffusion import AdvDiffusion

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

adv_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda().eval()

image_sample = torch.rand([1, 3, 128, 128]).cuda()
#
# print(image_sample)
#
print(model(image_sample, torch.tensor([30]).cuda()))
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from denoising_diffusion_pytorch import Unet
import argparse
from adversarial_diffusion.trainer import Trainer

from adversarial_diffusion.adv_diffusion import AdvDiffusion

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

adv_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda().eval()

# image_sample = torch.rand([1, 3, 128, 128]).cuda()
#
# print(image_sample)
#
# print(model(image_sample, torch.tensor([0]).cuda()))
#
# print(adv_model(image_sample, torch.tensor([0]).cuda()))

parser = argparse.ArgumentParser()
parser.add_argument("--adv", type=int, default=1)

args = parser.parse_args()


diffusion = AdvDiffusion(
    model,
    adv_model if args.adv == 1 else None,
    image_size=128,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference)
    loss_type="l1",  # L1 or L2
).cuda()

# data_dir = "../../../data/home/rainwangphy/images"
data_dir = "./images"
trainer = Trainer(
    diffusion,
    data_dir,
    train_batch_size=48,
    train_lr=8e-5,
    train_num_steps=50000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
)

trainer.train()

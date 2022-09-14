import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference)
    loss_type="l1",  # L1 or L2
).cuda()


data_dir = "../../../data/home/rainwangphy/images"
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

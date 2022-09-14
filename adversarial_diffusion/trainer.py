import math
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from adversarial_diffusion.dataset import DMDataset
from adversarial_diffusion.utils import (
    cycle,
    has_int_squareroot,
    exists,
    num_to_groups,
)

# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision="fp16" if fp16 else "no"
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = DMDataset(
            folder,
            (self.image_size, self.image_size),
            augment_horizontal_flip=augment_horizontal_flip,
            convert_image_to=convert_image_to,
        )
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count(),
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(batch_size=n),
                                    batches,
                                )
                            )

                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(
                            all_images,
                            str(self.results_folder / f"sample-{milestone}.png"),
                            nrow=int(math.sqrt(self.num_samples)),
                        )
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print("training complete")

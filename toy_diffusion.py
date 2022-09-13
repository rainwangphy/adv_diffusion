#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, argparse

import matplotlib.pyplot as plt

import torch, torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device {device}")

######################################################################


def sample_gaussian_mixture(nb):
    p, std = 0.3, 0.2
    result = torch.randn(nb, 1) * std
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result


def sample_ramp(nb):
    result = torch.min(torch.rand(nb, 1), torch.rand(nb, 1))
    return result


def sample_two_discs(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    q = (torch.rand(nb) <= 0.5).long()
    b = b * (0.3 + 0.2 * q)
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b - 0.5 + q
    result[:, 1] = a.sin() * b - 0.5 + q
    return result


def sample_disc_grid(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    N = 4
    q = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    r = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    b = b * 0.1
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b + q
    result[:, 1] = a.sin() * b + r
    return result


def sample_spiral(nb):
    u = torch.rand(nb)
    rho = u * 0.65 + 0.25 + torch.rand(nb) * 0.15
    theta = u * math.pi * 3
    result = torch.empty(nb, 2)
    result[:, 0] = theta.cos() * rho
    result[:, 1] = theta.sin() * rho
    return result


def sample_mnist(nb):
    train_set = torchvision.datasets.MNIST(root="./data/", train=True, download=True)
    result = train_set.data[:nb].to(device).view(-1, 1, 28, 28).float()
    return result


samplers = {
    "gaussian_mixture": sample_gaussian_mixture,
    "ramp": sample_ramp,
    "two_discs": sample_two_discs,
    "disc_grid": sample_disc_grid,
    "spiral": sample_spiral,
    "mnist": sample_mnist,
}

######################################################################

parser = argparse.ArgumentParser(
    description="""A minimal implementation of Jonathan Ho, Ajay Jain, Pieter Abbeel
"Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239""",
)

parser.add_argument(
    "--seed", type=int, default=0, help="Random seed, < 0 is no seeding"
)

parser.add_argument("--nb_epochs", type=int, default=100, help="How many epochs")

parser.add_argument("--batch_size", type=int, default=25, help="Batch size")

parser.add_argument(
    "--nb_samples", type=int, default=25000, help="Number of training examples"
)

parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

parser.add_argument(
    "--ema_decay", type=float, default=0.9999, help="EMA decay, <= 0 is no EMA"
)

data_list = ", ".join([str(k) for k in samplers])

parser.add_argument(
    "--data",
    type=str,
    default="spiral",
    help=f"Toy data-set to use: {data_list}",
)

args = parser.parse_args()

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.mem = {}
        with torch.no_grad():
            for p in model.parameters():
                self.mem[p] = p.clone()

    def step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                self.mem[p].copy_(self.decay * self.mem[p] + (1 - self.decay) * p)

    def copy_to_model(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(self.mem[p])


######################################################################


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ks, nc = 5, 64

        self.core = nn.Sequential(
            nn.Conv2d(in_channels, nc, ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding=ks // 2),
            nn.ReLU(),
            nn.Conv2d(nc, out_channels, ks, padding=ks // 2),
        )

    def forward(self, x):
        return self.core(x)


######################################################################
# Data

try:
    train_input = samplers[args.data](args.nb_samples).to(device)
except KeyError:
    print(f"unknown data {args.data}")
    exit(1)

train_mean, train_std = train_input.mean(), train_input.std()

######################################################################
# Model

if train_input.dim() == 2:
    nh = 256

    model = nn.Sequential(
        nn.Linear(train_input.size(1) + 1, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, train_input.size(1)),
    )

elif train_input.dim() == 4:

    model = ConvNet(train_input.size(1) + 1, train_input.size(1))

else:
    raise NotImplementedError

model.to(device)

print(f"nb_parameters {sum([ p.numel() for p in model.parameters() ])}")

######################################################################
# Generate


def generate(size, alpha, alpha_bar, sigma, model, train_mean, train_std):

    with torch.no_grad():

        x = torch.randn(size, device=device)

        for t in range(T - 1, -1, -1):
            z = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            input = torch.cat((x, torch.full_like(x[:, :1], t / (T - 1) - 0.5)), 1)
            x = (
                1
                / torch.sqrt(alpha[t])
                * (x - (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t]) * model(input))
                + sigma[t] * z
            )

        x = x * train_std + train_mean

        return x


######################################################################
# Train

T = 1000
beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1 - beta
alpha_bar = alpha.log().cumsum(0).exp()
sigma = beta.sqrt()

ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

for k in range(args.nb_epochs):

    acc_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for x0 in train_input.split(args.batch_size):
        x0 = (x0 - train_mean) / train_std
        t = torch.randint(T, (x0.size(0),) + (1,) * (x0.dim() - 1), device=x0.device)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * eps
        input = torch.cat((xt, t.expand_as(x0[:, :1]) / (T - 1) - 0.5), 1)
        loss = (eps - model(input)).pow(2).mean()
        acc_loss += loss.item() * x0.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.step()

    print(f"{k} {acc_loss / train_input.size(0)}")

if ema is not None:
    ema.copy_to_model()

######################################################################
# Plot

model.eval()

if train_input.dim() == 2:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Nx1 -> histogram
    if train_input.size(1) == 1:

        x = generate((10000, 1), alpha, alpha_bar, sigma, model, train_mean, train_std)

        ax.set_xlim(-1.25, 1.25)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        d = train_input.flatten().detach().to("cpu").numpy()
        ax.hist(
            d,
            25,
            (-1, 1),
            density=True,
            histtype="stepfilled",
            color="lightblue",
            label="Train",
        )

        d = x.flatten().detach().to("cpu").numpy()
        ax.hist(
            d,
            25,
            (-1, 1),
            density=True,
            histtype="step",
            color="red",
            label="Synthesis",
        )

        ax.legend(frameon=False, loc=2)

    # Nx2 -> scatter plot
    elif train_input.size(1) == 2:

        x = generate((1000, 2), alpha, alpha_bar, sigma, model, train_mean, train_std)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set(aspect=1)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        d = x.detach().to("cpu").numpy()
        ax.scatter(d[:, 0], d[:, 1], s=2.0, color="red", label="Synthesis")

        d = train_input[: x.size(0)].detach().to("cpu").numpy()
        ax.scatter(d[:, 0], d[:, 1], s=2.0, color="gray", label="Train")

        ax.legend(frameon=False, loc=2)

    filename = f"diffusion_{args.data}.pdf"
    print(f"saving {filename}")
    fig.savefig(filename, bbox_inches="tight")

    if hasattr(plt.get_current_fig_manager(), "window"):
        plt.get_current_fig_manager().window.setGeometry(2, 2, 1024, 768)
        plt.show()

# NxCxHxW -> image
elif train_input.dim() == 4:

    x = generate(
        (128,) + train_input.size()[1:],
        alpha,
        alpha_bar,
        sigma,
        model,
        train_mean,
        train_std,
    )
    x = 1 - x.clamp(min=0, max=255) / 255
    torchvision.utils.save_image(
        x, f"diffusion_{args.data}.png", nrow=16, pad_value=0.8
    )

######################################################################

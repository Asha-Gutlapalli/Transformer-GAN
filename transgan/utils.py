import os
import math
import random
import warnings
import subprocess
import numpy as np
from pathlib import Path
from random import random
from functools import partial
from datetime import datetime

# "contextlib" module provides utilities for working with context managers and "with" statements.
# - "contextmanager" from "contextlib" is a decorator that manages resources.
from contextlib import contextmanager

from einops import repeat, rearrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad


#Helper Functions and Classes

# returns current timestamp that is later used to name output files
def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

# checks if the given item exists
def exists(val):
    return val is not None

# casts to list
def cast_list(el):
    return el if isinstance(el, list) else [el]

# returns default value if the given value does not exist
def default(value, d):
    return value if exists(value) else d

# returns items from iterable
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# null context
@contextmanager
def null_context():
    yield

# raises Nan exception
def raise_if_nan(t):
    if torch.isnan(t):
        raise ValueError("")

# loss is propagated backwards
def loss_backwards(loss, **kwargs):
    loss.backward(**kwargs)

# model evalutes the batches in chunks
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

# applies a function at random
class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

# accumulates contexts periodically
def gradient_accumulate_contexts(gradient_accumulate_every):
    contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

# Gradient Penalty
# ----------------
# - Applies gradient penalty to ensure stability in GAN training by preventing exploding gradients in the discriminator.
# - Read about it at https://arxiv.org/pdf/1704.00028.pdf
# - Watch about it at https://www.youtube.com/watch?v=5c57gnaPkA4
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# normalization

# instance normalization
ChanNorm = partial(nn.InstanceNorm2d, affine = True)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, dim_context = None):
        super().__init__()
        self.norm = ChanNorm(dim)
        self.norm_context = ChanNorm(dim_context) if exists(dim_context) else None
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs.pop('context')
            context = self.norm_context(context)
            kwargs.update(context = context)

        return self.fn(x, **kwargs)

# upsample layer
def upsample(scale_factor = 2):
    return nn.Upsample(scale_factor = scale_factor)

# Leaky ReLU
# ----------
# - Leaky ReLU is an activation function that fixes the "dyling ReLU" problem - max(0.1x, x)
# - Read about it at https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24#0863
# - Watch a video about it at https://www.youtube.com/watch?v=Y-ruNSdpZ0Q
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

# rotary positional embedding helpers
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)     #[1, 1, D, D/2, 2]

    x1, x2 = x.unbind(dim = -1)                         #[1, 1, D, D/2]

    x = torch.stack((-x2, x1), dim = -1)                #[1, 1, D, D/2]

    return rearrange(x, '... d j -> ... (d j)')         #[1, 1, D, D]

def get_sin_cos(seq):
    n = seq.shape[0]

    x_sinu = repeat(seq, 'i d -> i j d', j = n)                                     #[H ** 0.5, H ** 0.5, D/4]
    y_sinu = repeat(seq, 'j d -> i j d', i = n)                                     #[H ** 0.5, H ** 0.5, D/4]

    sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)                         #[H ** 0.5, H ** 0.5, D/2]
    cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)                         #[H ** 0.5, H ** 0.5, D/2]

    sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))          #[H, D/2]
    sin, cos = map(lambda t: repeat(t, 'n d -> () () n (d j)', j = 2), (sin, cos))  #[1, 1, D, D]
    
    return sin, cos

# residual connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        if isinstance(out, tuple):
            out, latent = out
            ret = (out + x, latent)
            return ret

        return x + out

# adds the outputs of given functions together
class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))

# Hinge loss for generator
def gen_hinge_loss(fake, real):
    return fake.mean()

# Hinge loss for discriminator
def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

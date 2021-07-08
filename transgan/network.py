import math
from math import log, log2, sqrt, pi
from functools import partial
import multiprocessing

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn, einsum # "einsum" from "pytorch" allows various multi-dimensional tensor operations

'''
"einops" is a library that makes tensor operations easier.
-"rearrange" from "einops" reorders multi-dimensional tensors.
-"repeat" from "einops" reorders and repeats elements in arbitrary combinations.
'''
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from .utils import *
from .augment import AugWrapper

# Depth-wise Seperable Convolution
# - -------------------------------
# -  Depth-wise Seperable Convolution is a type of convolution that is much faster and requires less number of parameters.
# -  Read about it at https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec, https://arxiv.org/pdf/1610.02357.pdf
# -  Watch a video about it at https://www.youtube.com/watch?v=T7o3xvJLuHk

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# Gated Convolution Network
#---------------------
# - It combines convolution networks with a gating mechanism.
# - Read about it at https://arxiv.org/pdf/1612.08083.pdf

def FeedForward(dim, mult = 4, kernel_size = 3, bn = False):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(dim, dim * mult * 2, 1),                                              #[N, (dim * mult * 2), H, W]
        nn.GLU(dim = 1),                                                                #[N, (dim * mult * 2) / 2, H , W]
        nn.BatchNorm2d(dim * mult) if bn else nn.Identity(),                            #[N, (dim * mult), H, W]
        DepthWiseConv2d(dim * mult, dim * mult * 2, kernel_size, padding = padding),    #[N, (dim * mult * 2), H, W]
        nn.GLU(dim = 1),                                                                #[N, (dim * mult * 2) / 2, H , W]
        nn.Conv2d(dim * mult, dim, 1)                                                   #[N, dim, H, W]
    )

# Sinusoidal Positional Embedding
# - -----------------------------
# - Sinusoidal Postional Embedding encodes the position of the a word in a sentence.
# - Read about it at https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# - Watch a video about it at https://www.youtube.com/watch?v=dichIcUZfOw

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dim //= 2
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        h = torch.linspace(-1., 1., x.shape[-2], device = x.device).type_as(self.inv_freq)
        w = torch.linspace(-1., 1., x.shape[-1], device = x.device).type_as(self.inv_freq)

        sinu_inp_h = torch.einsum('i , j -> i j', h, self.inv_freq)          #[H, C/8]
        sinu_inp_w = torch.einsum('i , j -> i j', w, self.inv_freq)          #[W, C/8]

        sinu_inp_h = repeat(sinu_inp_h, 'h c -> () c h w', w = x.shape[-1])  #[B, C/4, H, W]
        sinu_inp_w = repeat(sinu_inp_w, 'w c -> () c h w', h = x.shape[-2])  #[B, C/4, H, W]

        sinu_inp = torch.cat((sinu_inp_w, sinu_inp_h), dim = 1)              #[B, C/2, H, W]

        emb = torch.cat((sinu_inp.sin(), sinu_inp.cos()), dim = 1)           #[B, C, H, W]

        return emb

# Rotary Position Embedding
#--------------------------
# - Rotary Position Embedding encodes absolute positional information with rotation matrix.
# - Read about it at https://arxiv.org/pdf/2104.09864.pdf

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, downsample_keys = 1):
        super().__init__()
        self.dim = dim
        self.downsample_keys = downsample_keys

    def forward(self, q, k):
        device, dtype, n = q.device, q.dtype, int(sqrt(q.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)                                                       #[H ** 0.5]
        seq = seq.unsqueeze(-1)                                                                                         #[H ** 0.5, 1]

        scales = torch.logspace(0., log(10 / 2) / log(2), self.dim // 4, base = 2, device = device, dtype = dtype)      #[D/4]
        scales = scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]                                                  #[1, D/4]

        seq = seq * scales * pi                                                                                         #[H ** 0.5, D/4]

        x = seq
        y = seq

        y = reduce(y, '(j n) c -> j c', 'mean', n = self.downsample_keys)                                               #[(H ** 0.5) / B, D/4]

        q_sin, q_cos = get_sin_cos(x)                                                                                   #[1, 1, D, D]
        k_sin, k_cos = get_sin_cos(y)                                                                                   #[1, 1, D, D]

        q = (q * q_cos) + (rotate_every_two(q) * q_sin)                                                                 #[1, 1, D, D]
        k = (k * k_cos) + (rotate_every_two(k) * k_sin)                                                                 #[1, 1, D, D]

        return q, k


# Bipartite Attention
# -------------------
# - Bipartite Attention is a generalization of self-attention.
# - But it is between two group of variables (image features and latents).
# - Read about it at https://arxiv.org/pdf/2103.01209.pdf

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size = None,
        dim_out = None,
        kv_dim = None,
        heads = 8,
        dim_head = 64,
        q_kernel_size = 1,
        kv_kernel_size = 3,
        out_kernel_size = 1,
        q_stride = 1,
        include_self = False,
        downsample = False,
        downsample_kv = 1,
        bn = False,
        latent_dim = None
    ):
        super().__init__()
        # positional embedding
        self.sinu_emb = FixedPositionalEmbedding(dim)

        # set up attention parameters
        inner_dim = dim_head *  heads
        kv_dim = default(kv_dim, dim)
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # padding for qkv
        q_padding = q_kernel_size // 2
        kv_padding = kv_kernel_size // 2
        out_padding = out_kernel_size // 2

        # convolution parameters for q
        q_conv_params = (1, 1, 0)

        # query
        self.to_q = nn.Conv2d(dim, inner_dim, *q_conv_params, bias = False)

        # convolution parameters for each downsample factor for k and v
        if downsample_kv == 1:
            kv_conv_params = (3, 1, 1) #(kernel_size, stride, padding)
        elif downsample_kv == 2:
            kv_conv_params = (3, 2, 1)
        elif downsample_kv == 4:
            kv_conv_params = (7, 4, 3)
        else:
            raise ValueError(f'invalid downsample factor for key / values {downsample_kv}')

        # key and value (context)
        self.to_k = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)
        self.to_v = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)

        # batch normalization for q, k and v
        self.bn = bn
        if self.bn:
            self.q_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.k_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.v_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()

        # latents
        self.has_latents = exists(latent_dim)
        if self.has_latents:
            # latent normalization
            self.latent_norm = ChanNorm(latent_dim)
            # latents to q, k, and v
            self.latents_to_qkv = nn.Conv2d(latent_dim, inner_dim * 3, 1, bias = False)
            # attention output - latents
            # feed forward network
            self.latents_to_out = nn.Sequential(
                nn.Conv2d(inner_dim, latent_dim * 2, 1),
                nn.GLU(dim = 1),
                nn.BatchNorm2d(latent_dim) if bn else nn.Identity()
            )

        # key and value (image features)
        self.include_self = include_self
        if include_self:
            self.to_self_k = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)
            self.to_self_v = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)

        # mixed heads from mixed multi-head attention
        self.mix_heads_post = nn.Parameter(torch.randn(heads, heads))

        # convolution paramters for attention output
        out_conv_params = (3, 2, 1) if downsample else q_conv_params

        # attention output
        # feed forward network
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out * 2, *out_conv_params),
            nn.GLU(dim = 1),
            nn.BatchNorm2d(dim_out) if bn else nn.Identity()
        )

        self.fmap_size = fmap_size
        # rotary embedding
        self.pos_emb = RotaryEmbedding(dim_head, downsample_keys = downsample_kv)

    def forward(self, x, latents = None, context = None, include_self = False):
        # check if image size is power of 2
        assert not exists(self.fmap_size) or x.shape[-1] == self.fmap_size, 'fmap size must equal the given shape'

        b, n, _, y, h, device = *x.shape, self.heads, x.device

        # check for context
        has_context = exists(context)
        context = default(context, x)

        q_inp = x
        k_inp = context
        v_inp = context

        # positional embedding
        if not has_context:
            sinu_emb = self.sinu_emb(context)
            q_inp += sinu_emb
            k_inp += sinu_emb

        # query, key, and value
        q, k, v = (self.to_q(q_inp), self.to_k(k_inp), self.to_v(v_inp))

        # batch normalization
        if self.bn:
            q = self.q_bn(q)
            k = self.k_bn(k)
            v = self.v_bn(v)

        out_h, out_w = q.shape[-2:]

        # split heads
        split_head = lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h)

        q, k, v = map(split_head, (q, k, v))

        # rotary embedding in the absence of context
        if not has_context:
            q, k = self.pos_emb(q, k)

        # key and value (image features)
        if self.include_self:
            kx = self.to_self_k(x)
            vx = self.to_self_v(x)
            kx, vx = map(split_head, (kx, vx))

            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

        # latents
        if self.has_latents:
            assert exists(latents), 'latents must be passed in'
            latents = self.latent_norm(latents)
            lq, lk, lv = self.latents_to_qkv(latents).chunk(3, dim = 1)
            lq, lk, lv = map(split_head, (lq, lk, lv))

            latent_shape = lq.shape
            num_latents = lq.shape[-2]

            q = torch.cat((lq, q), dim = -2)
            k = torch.cat((lk, k), dim = -2)
            v = torch.cat((lv, v), dim = -2)

        # Multi-Head Attention
        #---------------------
        # Self-Attention allows input to interact with eachother and captures the relation between them.
        # Self-Attention with multiple attention heads is Multi-Head Attention.
        # - Read about it at https://arxiv.org/pdf/1706.03762.pdf
        # - Watch a video about it at https://www.youtube.com/watch?v=iDulhoQ2pro

        # (QK.T)/sqrt(dim_K)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # softmax((QK.T)/sqrt(dim_K))
        attn = dots.softmax(dim = -1)
        # mixed heads
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post)
        # softmax((QK.T)/sqrt(dim_K))V
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # output and latent output
        if self.has_latents:
            lout, out = out[..., :num_latents, :], out[..., num_latents:, :]
            lout = rearrange(lout, 'b h (x y) d -> b (h d) x y', h = h, x = latents.shape[-2], y = latents.shape[-1])
            lout = self.latents_to_out(lout)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, x = out_h, y = out_w)
        out = self.to_out(out)

        if self.has_latents:
            return out, lout

        return out

# linear layer with learning rate
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

# mapping network
class MappingNetwork(nn.Module):
    def __init__(self, dim, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(
            *layers,
            nn.Linear(dim, dim * 4)
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.net(x)
        return rearrange(x, 'b (c h w) -> b c h w', h = 2, w = 2)

# ------------------------------------------------------------------------------------------------------------------------
# Generative Adversarial Network

# Generator
class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        fmap_max = 512,
        init_channel = 3,
        mapping_network_depth = 4
    ):
        super().__init__()
        # check if image size is power of 2
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        # number of layers
        num_layers = int(log2(image_size)) - 1

        # mapping network
        self.mapping = MappingNetwork(latent_dim, mapping_network_depth)
        # initial block
        self.initial_block = nn.Parameter(torch.randn((latent_dim, 4, 4)))
        # layers
        self.layers = nn.ModuleList([])

        # set up generator paramters
        fmap_size = 4
        chan = latent_dim
        min_chan = 8

        for ind in range(num_layers):
            # check if it is the last layer
            is_last = ind == (num_layers - 1)

            # downsample factor
            downsample_factor = int(2 ** max(log2(fmap_size) - log2(32), 0))
            # attention
            attn_class = partial(Attention, bn = True, fmap_size = fmap_size, downsample_kv = downsample_factor)

            # upsample layer
            if not is_last:
                chan_out = max(min_chan, chan // 4)

                # attention --> pixel shuffle
                upsample = nn.Sequential(
                    attn_class(dim = chan, dim_head = chan, heads = 1, dim_out = chan_out * 4),

                    # Pixel Shuffle
                    # -------------
                    # Pixel Shuffle produces high resolution images from low resolution images using a shuffling operation.
                    # Read about it at https://arxiv.org/pdf/1609.05158.pdf

                    nn.PixelShuffle(2)
                )

            else:
                upsample = nn.Identity()

            # attention --> feed forward network --> upsample
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(chan, attn_class(dim = chan, latent_dim = latent_dim))),
                Residual(FeedForward(chan, bn = True, kernel_size = (3 if image_size > 4 else 1))),
                upsample,
            ]))

            chan = chan_out
            fmap_size *= 2

        # final attention layer
        self.final_attn = Residual(PreNorm(chan, attn_class(chan, latent_dim = latent_dim)))

        # feed forward network --> final image
        self.to_img = nn.Sequential(
            Residual(FeedForward(chan_out, bn = True)),
            nn.Conv2d(chan, init_channel, 1)
        )

    def forward(self, x):
        b = x.shape[0]

        latents = self.mapping(x)

        fmap = repeat(self.initial_block, 'c h w -> b c h w', b = b)

        for attn, ff, upsample in self.layers:
            fmap, latents_out = attn(fmap, latents = latents)
            latents = latents + latents_out

            fmap = ff(fmap)
            fmap = upsample(fmap)

        fmap, _ = self.final_attn(fmap, latents = latents)
        return self.to_img(fmap)

# Discriminator
class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 256,
        init_channel = 3,
    ):
        super().__init__()
        # check if image size is power of 2
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        # number of layers
        num_layers = int(log2(image_size)) - 2
        fmap_dim = 64

        # embedding
        self.conv_embed = nn.Sequential(
            nn.Conv2d(init_channel, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv2d(32, fmap_dim, kernel_size = 3, padding = 1)
        )

        image_size //= 2

        # axial postional embedding
        self.ax_pos_emb_h = nn.Parameter(torch.randn(image_size, fmap_dim))
        self.ax_pos_emb_w = nn.Parameter(torch.randn(image_size, fmap_dim))

        # image sizes
        self.image_sizes = []
        # layers
        self.layers = nn.ModuleList([])
        # feature map dimensions
        fmap_dims = []

        for ind in range(num_layers):
            image_size //= 2

            self.image_sizes.append(image_size)

            fmap_dim_out = min(fmap_dim * 2, fmap_max)

            # add the original and downsampled feature maps
            downsample = SumBranches([
                nn.Conv2d(fmap_dim, fmap_dim_out, 3, 2, 1),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(fmap_dim, fmap_dim_out, 3, padding = 1),
                    leaky_relu()
                )
            ])

            #downsample factor
            downsample_factor = 2 ** max(log2(image_size) - log2(32), 0)
            # attention
            attn_class = partial(Attention, fmap_size = image_size, downsample_kv = downsample_factor)

            # downsample --> attention --> feed forward network
            self.layers.append(nn.ModuleList([
                downsample,
                Residual(PreNorm(fmap_dim_out, attn_class(dim = fmap_dim_out))),
                Residual(PreNorm(fmap_dim_out, FeedForward(dim = fmap_dim_out, kernel_size = (3 if image_size > 4 else 1))))
            ]))

            fmap_dim = fmap_dim_out
            fmap_dims.append(fmap_dim)

        # attention --> feed forward network --> convolution --> logits
        self.to_logits = nn.Sequential(
            Residual(PreNorm(fmap_dim, Attention(dim = fmap_dim, fmap_size = 2))),
            Residual(PreNorm(fmap_dim, FeedForward(dim = fmap_dim, kernel_size = (3 if image_size > 64 else 1)))),
            nn.Conv2d(fmap_dim, 1, 2),
            Rearrange('b () () () -> b')
        )

    def forward(self, x):
        x_ = x
        x = self.conv_embed(x)

        ax_pos_emb = rearrange(self.ax_pos_emb_h, 'h c -> () c h ()') + rearrange(self.ax_pos_emb_w, 'w c -> () c () w')
        x += ax_pos_emb

        fmaps = []

        for (downsample, attn, ff), image_size in zip(self.layers, self.image_sizes):
            x = downsample(x)
            x = attn(x)
            x = ff(x)

            fmaps.append(x)

        x = self.to_logits(x)

        return x, None

# transGANformer
class Transganformer(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        fmap_max = 512,
        transparent = False,
        ttur_mult = 1.,
        lr = 2e-4,
        device = "cuda"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        if transparent:
            init_channel = 4
        else:
            init_channel = 3

        # generator
        self.G = Generator(
            image_size = image_size,
            latent_dim = latent_dim,
            fmap_max = fmap_max,
            init_channel = init_channel)

        # discriminator
        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            init_channel = init_channel
        )

        # init optimizers
        self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights

        self.to(device)

        # wrapper for augmenting all images going into the discriminator - Adaptive Discriminator Augmentation
        self.D_aug = AugWrapper(self.D, image_size)

    # initializes weights
    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        raise NotImplemented

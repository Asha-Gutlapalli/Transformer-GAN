import json
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from math import floor
from pathlib import Path
from shutil import rmtree

import torchvision
from torch.utils import data
from torchvision import transforms

from .data import Dataset
from .network import *

# Constants

# number of CPU cores
NUM_CORES = multiprocessing.cpu_count()


# Training Code
class Trainer():
    def __init__(
        self,
        name = 'trippy',
        results_dir = 'samples',
        models_dir = 'models',
        base_dir = './',
        latent_dim = 256,
        image_size = 128,
        num_image_tiles = 8,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        gradient_accumulate_every = 1,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 1.,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        aug_prob = 0.25,
        aug_types = ['translation', 'cutout', 'color'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (32, 64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.num_image_tiles = num_image_tiles
        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulate_every = gradient_accumulate_every

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

    # Checks if images are transparent
    # JPG - Not transparent
    # PNG - Transparent
    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    # returns checkpoint number
    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    # initializes GAN
    def init_GAN(self):
        args, kwargs = self.GAN_params

        self.GAN = Transganformer(
            lr = self.lr,
            latent_dim = self.latent_dim,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            transparent = self.transparent,
            *args,
            **kwargs
        )

    # writes configuration to a config file
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    # loads configuration from a config file
    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    # returns configuration
    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
        }

    # loads dataset
    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES)
        sampler = None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = self.batch_size, sampler = sampler, shuffle = False, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')
    
    # trains model
    def train(self):
        # checks if dataloader exists
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        # initializes GAN if not already
        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()

        # initializes total generator and discriminator loss
        total_disc_loss = torch.zeros([], device=self.device)
        total_gen_loss = torch.zeros([], device=self.device)

        batch_size = self.batch_size

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        # setup augmentation arguments
        aug_prob   = default(self.aug_prob, 0)
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G
        D = self.GAN.D
        D_aug = self.GAN.D_aug

        # setup losses
        D_loss_fn = hinge_loss
        G_loss_fn = gen_hinge_loss

        # applies gradient penalty periodically
        apply_gradient_penalty = self.steps % 4 == 0

        # train discriminator

        # sets gradients of discriminator's paramters to zero to clear old gradients
        self.GAN.D_opt.zero_grad()
        
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):
            # get latents
            latents = torch.randn(batch_size, latent_dim).to(self.device)

            # generated images
            generated_images = G(latents)
            # output of generated images from dicriminator with augmentation
            fake_output, _ = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

            # load next batch of images
            image_batch = next(self.loader).to(self.device)
            image_batch.requires_grad_()

            # output of real images from dicriminator with augmentation
            real_output, real_aux_loss = D_aug(image_batch, **aug_kwargs)

            # discriminator loss function
            divergence = D_loss_fn(real_output, fake_output)
            disc_loss = divergence

            # applies gradient penalty and calculates gradient penalty loss
            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            # propagates loss backwards
            loss_backwards(disc_loss)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        # discriminator loss
        self.d_loss = float(total_disc_loss)

        # performs parameter update
        self.GAN.D_opt.step()


        # train generator

        # sets gradients of discriminator's paramters to zero to clear old gradients
        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):
            # get latents
            latents = torch.randn(batch_size, latent_dim).to(self.device)
            # generated images
            generated_images = G(latents)
            # output of generated images from dicriminator with augmentation
            fake_output, _ = D_aug(generated_images, **aug_kwargs)

            real_output = None

            # generator loss function
            loss = G_loss_fn(fake_output, real_output)
            gen_loss = loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            # propagates loss backwards
            loss_backwards(gen_loss)

            total_gen_loss += loss.item() / self.gradient_accumulate_every

        # generator loss
        self.g_loss = float(total_gen_loss)

        # performs parameter update
        self.GAN.G_opt.step()

        # save from NaN errors
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)

        # saves model checkpoint periodically
        if self.steps % self.save_every == 0:
            self.save(self.checkpoint_num)
        
        # saves intermediate results periodically
        if (self.steps % self.evaluate_every == 0) or (self.steps % 10 == 0):
            self.evaluate(floor(self.steps / self.evaluate_every))

        # calculates fid
        if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
            num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
            fid = self.calculate_fid(num_batches)
            self.last_fid = fid

            with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                f.write(f'{self.steps},{fid}\n')

        self.steps += 1
    
    # evaluates model
    @torch.no_grad()
    def evaluate(self, num = 0):
        self.GAN.eval()

        ext = self.image_extension

        # image grid side dimension
        num_rows = self.num_image_tiles
    
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents
        latents = torch.randn((num_rows ** 2, latent_dim)).to(self.device)

        # generates images and save image grid
        generated_images = self.generate_(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)


    # calculates fid
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.to(self.device).empty_cache()

        # setup paths to save fid scores for real and fake images
        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # removes any existing files used for fid calculation and recreate directories
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            # calculates fid for real images
            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    ind = k + batch_num * self.batch_size
                    torchvision.utils.save_image(image, real_path / f'{ind}.png')

        # generates a bunch of fake images in results / name / fid_fake
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()

        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # calculates fid for generated images
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents
            latents = torch.randn(self.batch_size, latent_dim).to(self.device)
            # generated images
            generated_images = self.generate_(self.GAN.G, latents)
            # saves images
            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                torchvision.utils.save_image(image, str(fake_path / f'{str(ind)}.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)

    # generates images from latent space
    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    # generates images from interpolation
    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, num_steps = 100, save_frames = False):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents
        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        # generates images from interpolated latents
        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_(self.GAN.G, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # saves frames
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))
    
    # prints out log
    def print_log(self):
        data = [
            ('G', self.g_loss),             # generator loss
            ('D', self.d_loss),             # discriminator loss
            ('GP', self.last_gp_loss),      # gradient penalty loss
            ('FID', self.last_fid)          # last fid score
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)
    
     # returns path to save model
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')
    
    # creates folders for saving models and results if not already
    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
    
    # removes existing folders for saving models, results, fid scores, and configurations
    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    # saves model checkpoint and configuration
    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    # loads model from model checkpoint
    def load(self, num=-1, print_version=True):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

            if len(saved_nums) == 0:
                return

            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
            
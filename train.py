# fire is a python library that automatically generates Command Line Interfaces (CLIs) from any python component
import fire
from tqdm import tqdm

import torch

from transgan.utils import cast_list, timestamped_filename
from transgan.train import Trainer

# main function
def train_from_folder(
    data = './Trippy_Image_Dataset',
    results_dir = './results',
    models_dir = './models',
    name = 'trippy',
    new = False,
    load_from = -1,
    image_size = 128,
    fmap_max = 512,
    transparent = False,
    batch_size = 1,
    gradient_accumulate_every = 120,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 100,
    evaluate_every = 100,
    generate = False,
    generate_interpolation = False,
    aug_prob=0.3,
    aug_types=['cutout', 'translation', 'color'],
    dataset_aug_prob=0.6,
    interpolation_num_steps = 100,
    save_frames = True,
    num_image_tiles = 8,
    num_workers = None,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
):

    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        num_image_tiles = num_image_tiles,
        num_workers = num_workers,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache
    )

    # generates sample images
    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    # generates images from interpolation
    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    model = Trainer(**model_args)

    # loads model from previous checkpoint unless specified otherwise
    if not new:
        model.load(load_from)
    else:
        model.clear()

    # loads dataset
    model.set_data_src(data)

    # trains model and prints log periodically
    for _ in tqdm(range(num_train_steps - model.steps), initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>'):
        model.train()

        if _ % 50 == 0:
            model.print_log()

    # saves model checkpoint
    model.save(model.checkpoint_num)

if __name__ == '__main__':
  # Fire exposes the contents of the program to the command line
  fire.Fire(train_from_folder)

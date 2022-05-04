import torch
import torchvision.transforms as transforms
import PIL
import os
import numpy as np
from tqdm import tqdm
import argparse

from prg.networks import DCGANDiscriminator128, DCGANGenerator128



parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_images', 
    '-n',
    help='Amount of images to generate',
    default=1
)
parser.add_argument(
    '--device',
    '-d',
    help='Device',
    default='cpu'
)
parser.add_argument(
    '--use_dropout',
    help='If true uses dropout else not',
    default=True, 
    choices=[True, False],
)


def make_image_out():
    if not os.path.isdir('image_out'):
        os.mkdir('image_out')


def load_model(device):
    try:
        generator = torch.load(
            './models/generator_200eps_s128_batch128_frac1', 
            map_location=device
        )
    except Exception as e:
        raise e('COuld not load model from folder ./models')
        
    return generator


def generate_images(generator, N, device, use_dropout):
    Z = torch.randn(int(N), 100, 1, 1, device=device)

    if use_dropout == 'False':
        with torch.no_grad():
            image_batch = generator(Z)
    else:
        image_batch = generator(Z)

    inverse_transforms = transforms.Compose(
        [
            transforms.Normalize(mean=(0, 0, 0), std=(1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize(mean=(-.5, -.5, -.5), std=(1, 1, 1)),
            transforms.ToPILImage()
        ]
    )  
    for i, image in enumerate(1, image_batch):
        image = inverse_transforms(image)
        image.save(f'./image_out/{i}.jpg')

   
def main(args):
    print('-- Lookup Folder ...')
    make_image_out()
    
    print('-- Load Model ...')
    generator = load_model(
        device=args.device
    )

    print('-- Generate Images ...')
    generate_images(
        generator=generator,
        N=args.n_images, 
        device=args.device, 
        use_dropout=args.use_dropout
    )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
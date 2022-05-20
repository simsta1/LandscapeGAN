import torch
import torchvision.transforms as transforms
import os
import argparse
import cv2

from prg.networks import DCGANDiscriminator128, DCGANGenerator128
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


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
    choices=['True', 'False'],
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

    return image_batch


def save_image_batch(image_batch):

    inverse_transforms = transforms.Compose(
        [
            transforms.Normalize(mean=(0, 0, 0), std=(1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize(mean=(-.5, -.5, -.5), std=(1, 1, 1)),
            transforms.ToPILImage()
        ]
    )  

    image_names = []
    for i, image in enumerate(image_batch):
        image = inverse_transforms(image)
        save_path = f'./image_out/{i+1}.jpg' 
        image.save(save_path)
        image_names.append(save_path)

def upscale_images():
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=4
    )
    netscale = 4
    model_path = './models/RealESRGAN_x4plus.pth' 

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False)

    # do upsampling using realsesrgan
    out_folder = 'image_out'
    images_to_upscale = [os.path.join(out_folder, name) for name in os.listdir(out_folder) if '.jpg' in name]
    images_to_upscale = [name for name in images_to_upscale if 'upscaled' not in name]
    for path in images_to_upscale:
        print(f'--- Upscaling Image: {path}')
        # laod image 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=4)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            extension = '.jpg'
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            path = path.split('.')[0]
            save_path = path + '_ups' + extension
            cv2.imwrite(save_path, output)
    

def main(args):
    print('-- Lookup Folder ...')
    make_image_out()
    
    print('-- Load Model ...')
    generator = load_model(
        device=args.device
    )

    print('-- Generate Images ...')
    image_batch = generate_images(
        generator=generator,
        N=args.n_images, 
        device=args.device, 
        use_dropout=args.use_dropout
    )
    print('-- Save Generated Images ...')
    save_image_batch(image_batch)

    print('-- Upscale Images ...')
    image_batch = upscale_images()


if __name__ == '__main__':
    main(args=parser.parse_args())
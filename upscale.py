import os
import argparse
import cv2

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', 
    '-i',
    help='Input folder for upscaling',
    default='image_out'
)
parser.add_argument(
    '--remove', 
    '-r',
    help='Remove not upscaled photos',
    default='True',
    choices=['True', 'False']
)


def upscale_images(input_folder):
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=4
    )
    model_path = './models/RealESRGAN_x4plus.pth' 

    # restorer
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False)

    # do upsampling using realsesrgan
    images_to_upscale = [os.path.join(input_folder, name) for name in os.listdir(input_folder) if '.jpg' in name]
    #images_to_upscale = [name for name in images_to_upscale if 'ups' not in name]
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

def remove_photos(input_folder):
    for path in [f for f in os.listdir(input_folder) if 'ups' not in f]:
        path = os.path.join(input_folder, path)
        if os.path.isfile:
            os.remove(path)

    

def main(args):

    print('-- Upscale Images ...')
    upscale_images(input_folder=args.input)

    if args.remove == 'True':
        print('-- Remove not upscaled images ...')
        remove_photos(input_folder=args.input)



if __name__ == '__main__':
    main(args=parser.parse_args())
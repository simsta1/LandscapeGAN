import PIL
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from multiprocessing import Pool

def __copy_images(image_paths: str):
    try:
        image = PIL.Image.open(image_paths[0])
        if image.size == (1024, 1024):
            os.rename(image_paths[0], image_paths[1])
        pbar.update(1)
    except:
        print(f'Could not move for {image_paths[0]} to {image_paths[1]}')
    
    
if __name__ == '__main__':
    
    # Params
    IMAGE_DIR = '../unsplash/'
    NEW_DIR = '../lhq_1024_jpg/'
    N_PROCESSES = 5
    PIL.Image.MAX_IMAGE_PIXELS = None
    
    # Create iterable for MP
    all_images = os.listdir(IMAGE_DIR)
    curr_paths = [os.path.join(IMAGE_DIR, img) for img in all_images]
    new_paths = [os.path.join(NEW_DIR, img) for img in all_images]
    mp_paths = list(zip(curr_paths, new_paths))
    
    # Initiate Progress bar
    pbar = tqdm(total=len(all_images), desc=f'Copying files from {IMAGE_DIR} to {NEW_DIR}')
    
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(__copy_images, mp_paths)
            
    
    
    
    
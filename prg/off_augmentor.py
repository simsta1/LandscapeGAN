import PIL
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from multiprocessing import Pool

def __transform_image(image_path: str):
    """Transforms one image given the transforms as Compose"""
    image = PIL.Image.open(image_path)
    if not image.size == (1024, 1024):
        crop_size = min(image.size)
        image_transformations = transforms.Compose([transforms.CenterCrop(crop_size), 
                                                    transforms.Resize(1024)])
        image = image_transformations(image)
        image.save(image_path)
    pbar.update(1)
    
    
if __name__ == '__main__':
    # Params
    IMAGE_DIR = '../unsplash/'
    N_PROCESSES = 5
    PIL.Image.MAX_IMAGE_PIXELS = None
    
    # Create Iterable for MP
    all_images = os.listdir(IMAGE_DIR)
    all_images = [os.path.join(IMAGE_DIR, img) for img in all_images]
    
    # INitiate Pbar
    pbar = tqdm(total=len(all_images))
    
    # Call MP
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(__transform_image, all_images)
            
    
    
    
    
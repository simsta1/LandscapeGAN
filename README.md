<a href="https://img.shields.io/badge/FHNW-Deep%20Learning-yellow"><img src="https://img.shields.io/badge/FHNW-Deep%20Learning-yellow" /></a>

# LandscapeGAN


<img src="doc\LandscapeGAN_gif.gif" width="600" align="center">

---


This repository is for the implementation of GAN (adapted DCGAN), which is able to sample Landscape Images from a random noise vector.

`Z -> DCGAN -> R-ESRGAN -> OUTPUT_IMAGE`

Repo borrows code for upscaling from https://github.com/xinntao/Real-ESRGAN

## Documentation

The documentation of the whole procedure from start to end is documented in german language within the notebook `LandsGAN.ipynb`

## Installation

1. Clone project locally 

```bash
git clone git@github.com:SimonStaehli/LandsGAN.git
```

2. Download Generator model from Google Drive and add it to the folder `./models` here: https://drive.google.com/drive/folders/13-GNHs1Bvjd17odvGciSg5K3eubKNEMd?usp=sharing 


3. Install required packages in this order
```bash
pip install -r requirements1.txt
pip install -r requirements2.txt
```

## Usage/Examples

### Generate images:

Will generate `5 photos` with size of 3x128x128 and place it into the folder image_out. Check parameter `--help` for more options and descriptions.

```bash
python runner.py -n 5
```

### Upscale images:

This script will upscale images within given `input folder -i` by 4 times. Check parameter `--help` for more options and descriptions.

```bash
python upscale.py -i image_out -r True
```

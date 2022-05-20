<a href="https://img.shields.io/badge/FHNW-Deep%20Learning-yellow"><img src="https://img.shields.io/badge/FHNW-Deep%20Learning-yellow" /></a>

# LandscapeGAN

This repository is for the implementation of GAN (adapted DCGAN), which is able to sample Landscape Images from a random noise vector.

`Z -> DCGAN -> R-ESRGAN -> OUTPUT_IMAGE`

Repo borrows code for upscaling from https://github.com/xinntao/Real-ESRGAN

## Documentation

The documentation of the whole procedure from start to end is documented within the notebook `LandsGAN.ipynb`

## Installation

1. Clone project locally 

```bash
git clone git@github.com:SimonStaehli/LandsGAN.git
```

2. Download Generator model from Google Drive and add it to the folder `./models` here: https://drive.google.com/drive/folders/13-GNHs1Bvjd17odvGciSg5K3eubKNEMd?usp=sharing 


3. Install required packages
```bash
pip install -r requirements.txt
```

## Usage/Examples

Basic Usage:
Will generate one photo and place it into the folder image_out.

```bash
python runner.py 
```

Additional Usage with params:

```bash
python runner.py --help
```

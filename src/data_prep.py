from __future__ import print_function, unicode_literals, absolute_import, division

import os

import numpy as np
import matplotlib
# matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
from stardist.matching import matching_dataset

np.random.seed(42)
lbl_cmap = random_label_cmap()

def check_sample_imgs():
    download_and_extract_zip_file(
        url='https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip',
        targetdir='data',
        verbose=1,
    )
    X = sorted(glob('data/dsb2018/train/images/*.tif'))
    Y = sorted(glob('data/dsb2018/train/masks/*.tif'))
    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))


def main():
    # check_sample_imgs()
    masks_path = 'data/masks/'

    X = sorted(glob('data/images/*.tif'))
    Y = sorted(glob('data/masks/*.tif'))
    print(X)
    print(Y)
    XY = zip(X, Y)
    # for x, y in XY:
    #     y_new = masks_path + x.split('/')[-1]
    #     print(y, y_new)
    #     os.rename(y, y_new)

    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))


if __name__ == '__main__':
    main()
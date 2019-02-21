import argparse
import os
import imageio

import numpy as np

from style_transfer import load_image, save_image, style_transfer
from utils import download_file_from_google_drive

STYLE_IDS = {
    'udnie': '0B9jhaT37ydSyb0NuYmk2ZEpOR0E',
    'wreck': '0B9jhaT37ydSySjNrM3J5N2gweVk',
    'wave': '0B9jhaT37ydSyVGk0TC10bDF1S28',
    'scream': '0B9jhaT37ydSyZ0RyTGU0Q2xiU28',
    'rain-princess': '0B9jhaT37ydSyaEJlSFlIeUxweGs',
    'la-muse': '0B9jhaT37ydSyQU1sYW02Sm9kV3c'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="Path to input image")
    parser.add_argument('--style', help="Which style to apply", choices=list(STYLE_IDS.keys()))
    parser.add_argument('--outdir', help="Path to output directory")
    parser.add_argument('--min-image-dim', help="Minimum image dimension", default=1000, type=int)
    parser.add_argument('--num-images', help="Number of images", default=50, type=int)
    parser.add_argument('--shimmer', help="Amount of movement", default=10, type=int)
    args = parser.parse_args()

    print("Loading image")
    img = load_image(args.image, args.min_image_dim)
    print("Downloading model")
    checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.ckpt' % args.style)
    download_file_from_google_drive(STYLE_IDS[args.style], checkpoint_path)

    noise1 = args.shimmer * np.random.uniform(size=img.shape) + 177
    noise2 = args.shimmer * np.random.uniform(size=img.shape) + 177

    with imageio.get_writer(os.path.join(args.outdir, 'output.gif'), mode='I') as writer:
        for i in range(args.num_images):
            mult = np.sin(i * 2 * np.pi / args.num_images) / 2 + 0.5
            noise = mult * noise1 + (1 - mult) * noise2
            input_img = img + noise
            print("Transferring style (%d/%d)" % (i+1, args.num_images))
            out = style_transfer(input_img, checkpoint_path)
            writer.append_data(out)
            print("Image saved")


if __name__ == '__main__':
    main()
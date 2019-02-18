import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from utils import download_file_from_google_drive
from utils import net

STYLE_IDS = {
    'udnie': '0B9jhaT37ydSyb0NuYmk2ZEpOR0E',
    'wreck': '0B9jhaT37ydSySjNrM3J5N2gweVk',
    'wave': '0B9jhaT37ydSyVGk0TC10bDF1S28',
    'scream': '0B9jhaT37ydSyZ0RyTGU0Q2xiU28',
    'rain-princess': '0B9jhaT37ydSyaEJlSFlIeUxweGs',
    'la-muse': '0B9jhaT37ydSyQU1sYW02Sm9kV3c'
}


def load_image(input_path, min_image_dim):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while min(img.shape[:2]) > min_image_dim:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return img


def save_image(img, output_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, 'output.jpg'), img)


def style_transfer(img, checkpoint_path):
    batch_size = 1
    batch_shape = (batch_size,) + img.shape
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        X = np.zeros(batch_shape, dtype=np.float32)
        X[0] = img

        _preds = sess.run(preds, feed_dict={img_placeholder:X})
        return _preds[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="Path to input image")
    parser.add_argument('--style', help="Which style to apply", choices=list(STYLE_IDS.keys()))
    parser.add_argument('--outdir', help="Path to output directory")
    parser.add_argument('--min-image-dim', help="Minimum image dimension", default=1000, type=int)
    args = parser.parse_args()

    print("Loading image")
    img = load_image(args.image, args.min_image_dim)
    print("Downloading model")
    checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.ckpt' % args.style)
    download_file_from_google_drive(STYLE_IDS[args.style], checkpoint_path)
    print("Transferring style")
    out = style_transfer(img, checkpoint_path)
    save_image(out, args.outdir)
    print("Image saved to %s" % os.path.join(args.outdir, 'output.jpg'))


if __name__ == '__main__':
    main()

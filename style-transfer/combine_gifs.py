import argparse
import os

import cv2
import imageio
import numpy as np

O_IMAGE_SIZE = (1008, 756)
N_IMAGE_SIZE = (336, 252)
C_SIZE = (672, 756)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gifs', help="Path to gif folder")
    parser.add_argument('--outdir', help="Path to output directory")
    args = parser.parse_args()

    readers = []
    for file in os.listdir(args.gifs):
        path = os.path.join(args.gifs, file)
        readers.append(imageio.get_reader(path))

    outpath = os.path.join(args.outdir, 'output.gif')
    writer = imageio.get_writer(outpath, mode='I')
    while True:
        try:
            frames = [reader.get_next_data() for reader in readers]
        except IndexError:
            break

        new_frame = np.zeros((C_SIZE[0], C_SIZE[1], 3)).astype(np.uint8)
        for i, frame in enumerate(frames):
            row = i // 3
            col = i % 3
            frame = cv2.resize(frame[:, :, :3], (N_IMAGE_SIZE[1], N_IMAGE_SIZE[0])).astype(np.uint8)
            new_frame[row * N_IMAGE_SIZE[0]:(row + 1) * N_IMAGE_SIZE[0], col * N_IMAGE_SIZE[1]:(col + 1) * N_IMAGE_SIZE[1], :] = frame

        writer.append_data(new_frame)

    writer.close()

if __name__ == '__main__':
    main()

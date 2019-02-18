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

BATCH_SIZE = 4


def style_transfer_video(video_file, checkpoint_path, out_path):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_size = (width // 2, height // 2)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, vid_size)

    g = tf.Graph()
    batch_shape = (BATCH_SIZE, vid_size[1], vid_size[0], 3)

    with g.as_default(), tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for j in range(count, BATCH_SIZE):
                X[j] = X[count - 1]

            _preds = sess.run(preds, feed_dict={img_placeholder: X})

            for j in range(count):
                style_frame = np.clip(_preds[j], 0, 255).astype(np.uint8)
                style_frame = cv2.cvtColor(style_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(style_frame)

        frame_count = 0
        i = 0
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            X[frame_count] = frame
            frame_count += 1
            if frame_count == BATCH_SIZE:
                style_and_write(frame_count)
                frame_count = 0
                print("Wrote %d frames" % (i + 1))
            i += 1

        if frame_count != 0:
            style_and_write(frame_count)

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help="Path to input video")
    parser.add_argument('--style', help="Which style to apply", choices=list(STYLE_IDS.keys()))
    parser.add_argument('--outdir', help="Path to output directory")
    parser.add_argument('--min-image-dim', help="Minimum image dimension", default=1000, type=int)
    args = parser.parse_args()

    print("Downloading model")
    checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.ckpt' % args.style)
    download_file_from_google_drive(STYLE_IDS[args.style], checkpoint_path)
    print("Transferring style")
    out_path = os.path.join(args.outdir, 'output.avi')
    style_transfer_video(args.video, checkpoint_path, out_path)
    print("Video saved to %s" % out_path)


if __name__ == '__main__':
    main()

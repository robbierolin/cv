import argparse
import os

import cv2
import numpy as np

from panorama import panorama


def get_frames_from_video(video_file):
    """
    Get frames from a video that can be stitched into a panorama.
    """
    cap = cv2.VideoCapture(video_file)

    feature_params = {
        'maxCorners': 100,
        'qualityLevel': 0.3,
        'minDistance': 7,
        'blockSize': 7
    }

    lk_params = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    colours = np.random.randint(0, 255, (100, 3))
    ret, old_frame = cap.read()

    # Get initial features to track.
    prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    num_points_start = len(p0)
    centroid = np.mean(p0, axis=0)

    # Create mask for drawing purposes
    mask = np.zeros_like(old_frame)

    # Frames to store for panorama.
    frames = [old_frame]
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow.
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        # Keep good points.
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # If number of points has dropped by > 80% or the centroid has moved over 20% of the image, take another image.
        centroid_cur = np.mean(p1, axis=0)
        if len(p1) < (num_points_start / 5) or np.linalg.norm(centroid - centroid_cur) > min(frame.shape[:2]) / 5:
            frames.append(frame)

            # Get new points to track.
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            num_points_start = len(p0)
            centroid = np.mean(p0, axis=0)
            mask = np.zeros_like(old_frame)
        else:
            p0 = good_new.reshape(-1, 1, 2)

        # Draw tracked points.
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), colours[i].tolist(), 2)

        img = cv2.add(frame, mask)

        # Show.
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        prev_gray = frame_gray.copy()

    cv2.destroyAllWindows()
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help="Path to input video")
    parser.add_argument('--outdir', help="Path to output directory")
    args = parser.parse_args()

    frames = get_frames_from_video(args.video)
    out_path = os.path.join(args.outdir, 'pano.jpg')
    panorama(frames, out_path)


if __name__ == '__main__':
    main()

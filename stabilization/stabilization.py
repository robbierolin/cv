import argparse
import os

import cv2
import numpy as np

SMOOTHING_RADIUS = 10


def _moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def _smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = _moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def _fix_border(frame):
    h, w, _ = frame.shape
    # Scale the image 4% from centre.
    T = cv2.getRotationMatrix2D((w/2, h/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame


def get_frame_transforms(cap):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _, prev = cap.read()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        p0 = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)

        _, frame = cap.read()

        if frame is None:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow.
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None)

        # Keep good points.
        p1 = p1[st == 1]
        p0 = p0[st == 1]

        # Find transformation matrix.
        m = cv2.estimateRigidTransform(p0, p1, fullAffine=False)

        # Translation.
        dx = m[0, 2]
        dy = m[1, 2]

        # Rotation angle.
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation.
        transforms[i] = [dx, dy, da]

        prev_gray = frame_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(p0)))

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = _smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference
    return transforms_smooth


def stabilize_and_write(cap, transforms, out_path):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 2):
        _, frame = cap.read()

        if frame is None:
            break

        dx, dy, da = transforms[i, :]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        frame_stabilized = _fix_border(frame_stabilized)

        frame_out = cv2.hconcat([frame, frame_stabilized])

        if frame_out.shape[1] > 1920:
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(1)
        out.write(frame_out)

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help="Path to video file")
    parser.add_argument('--outdir', help="Output directory")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    transforms = get_frame_transforms(cap)

    out_path = os.path.join(args.outdir, 'stabilized.mp4')

    stabilize_and_write(cap, transforms, out_path)


if __name__ == '__main__':
    main()

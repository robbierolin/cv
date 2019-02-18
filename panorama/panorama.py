import argparse
import os

import cv2
import imutils
import numpy as np


def load_images(input_dir):
    images = []
    for file_ in sorted(os.listdir(input_dir))[5:10]:
        if file_.endswith('.jpg'):
            path = os.path.join(input_dir, file_)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            images.append(img)

    return images


def panorama(images, output_path, crop=True):
    stitcher = cv2.createStitcher(cv2.STITCHER_SCANS)
    (status, stitched) = stitcher.stitch(images)

    if status == 1:
        print("Need more images")
        return
    elif status == 2:
        print("Failed to estimate homography")
        return
    elif status == 3:
        print("Failed to estimate camera parameters")
        return

    if crop:
        # Create a 10 pixel border surrounding the stitched image.
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

        # Convert to grayscale and threshold foregound.
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Find all external contours, then find the largest.
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        mask = np.zeros(thresh.shape, dtype='uint8')
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        min_rect = mask.copy()
        sub = mask.copy()

        # Grow mask until it non-zeros pixels are removed.
        while cv2.countNonZero(sub) > (max(stitched.shape) / 10):
            min_rect = cv2.erode(min_rect, None)
            sub = cv2.subtract(min_rect, thresh)

        cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        stitched = stitched[y:y+h, x:x+w]
    cv2.imwrite(output_path, stitched)
    print("Panorama saved to %s" % output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help="Path to folder of input images.")
    parser.add_argument('--outdir', help="Path to output directory.")
    parser.add_argument('--crop', help="Crop the image to remove stitching region", action='store_true', default=True)
    args = parser.parse_args()

    output_path = os.path.join(args.outdir, 'pano.jpg')

    images = load_images(args.images)
    panorama(images, output_path, args.crop)


if __name__ == '__main__':
    main()

# import the necessary packages
import apriltag
import argparse
import cv2
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*tif', '*tiff']:
        imgs.extend(sorted(glob(os.path.join(path, ext))))
    return imgs

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
        help="folder to input image containing AprilTag")
    args = vars(ap.parse_args())

    edge_length_thresh = 50

    # load the input image and convert it to grayscale
    image_dir = args["dir"]
    paths = glob_imgs(image_dir)
    # print(paths)
    mkdir_ifnotexists(os.path.join(image_dir, "output"))
    pointss = []
    image_width, image_height = 0, 0
    for path, count in zip(paths, range(len(paths))):
        families = ['tag36h10', 'tag36h11', 'tag25h9'] 
        # if count < 65:
        #     families = ['tag36h10', 'tag25h9'] 
        # else:
        #     families = ['tag36h11', 'tag25h9'] 
        pointss.append([])
        print("[INFO] loading image...")
        image = cv2.imread(path)
        image = cv2. cv2.bilateralFilter(image, 5, 60, 60)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)

        color_correction = False 
        if color_correction:
            gray = gray.astype(np.float32) / 255
            gray = gray * 1.2
            # gamma = 1.0 / 2.2
            # gray = np.power(gray,gamma)
            gray = (gray * 255.0).astype(np.uint8)
            cv2.imshow('gray_gamma_correction', gray)
            cv2.waitKey(0)

        image_height = gray.shape[0]
        image_width = gray.shape[1]

        # define the AprilTags detector options and then detect the AprilTags
        # in the input image
        results = []
        print("[INFO] detecting AprilTags...")
        for family in families:
            options = apriltag.DetectorOptions(families=family,
                                               border=0,
                                               nthreads=8,
                                               quad_decimate=2.0,
                                               quad_blur=1.0,
                                               refine_edges=True,
                                               refine_decode=False,
                                               refine_pose=False,
                                               debug=False,
                                               quad_contours=True)
            detector = apriltag.Detector(options)
            result = detector.detect(gray)
            results.append(result)
            print("[INFO] {} total AprilTags detected".format(len(result)))

        # loop over the AprilTag detection results
        for result, count in zip(results, range(len(results))):
            for r in result:
                # extract the bounding box (x, y)-coordinates for the AprilTag
                # and convert each of the (x, y)-coordinate pairs to integers
                (ptA, ptB, ptC, ptD) = r.corners
                edge_length = np.linalg.norm(ptA - ptC)
                if edge_length < edge_length_thresh:
                    continue
                ptB_int = (int(ptB[0]), int(ptB[1]))
                ptC_int = (int(ptC[0]), int(ptC[1]))
                ptD_int = (int(ptD[0]), int(ptD[1]))
                ptA_int = (int(ptA[0]), int(ptA[1]))
                # draw the bounding box of the AprilTag detection
                cv2.line(image, ptA_int, ptB_int, (0, 255, 0), 8)
                cv2.line(image, ptB_int, ptC_int, (0, 255, 0), 8)
                cv2.line(image, ptC_int, ptD_int, (0, 255, 0), 8)
                cv2.line(image, ptD_int, ptA_int, (0, 255, 0), 8)
                # draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
                # draw the tag family on the image
                # tagFamily = r.tag_family.decode("utf-8")
                tag_id = r.tag_id + 1000 * count 
                tagFamily = str(tag_id)
                temp = np.concatenate((np.array([tag_id]), ptA, ptB, ptC, ptD))
                pointss[-1].append(temp)
                cv2.putText(image, tagFamily, (int(ptA[0]), int(ptA[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                print("[INFO] tag family: {}".format(tagFamily))

        output_path = os.path.join(image_dir, "output", path.split('/')[-1])
        cv2.imwrite(output_path, image)

    print(pointss)
    fo = open(os.path.join(image_dir, "control_points.csv"), "w")
    for points, path in zip(pointss, paths):
        for point in points:
            for i in range(4):
                fo.write(path)
                fo.write(", ")
                fo.write("point_" + str(4 * int(point[0]) + i))
                fo.write(", ")
                point[2 * i + 1] = clamp(point[2 * i + 1], 0.0, float(image_width - 1))
                point[2 * i + 2] = clamp(point[2 * i + 2], 0.0, float(image_height - 1))

                fo.write(str(point[2 * i + 1]))
                fo.write(", ")
                fo.write(str(point[2 * i + 2]))
                fo.write("\n")
    fo.close()
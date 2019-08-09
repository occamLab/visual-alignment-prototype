#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys


# 2019-07-29-14-45-04
PATH = 'data/Documents/' + sys.argv[1] + '/'
IMAGE1 = cv2.imread(PATH + 'align-image.png', 0)
IMAGE2 = cv2.imread(PATH + 'camera-image.png', 0)
IMAGE1 = cv2.rotate(IMAGE1, cv2.ROTATE_90_CLOCKWISE)
IMAGE2 = cv2.rotate(IMAGE2, cv2.ROTATE_90_CLOCKWISE)
TEST_VECTORS1 = np.genfromtxt(PATH + 'vectors1.txt')
TEST_VECTORS2 = np.genfromtxt(PATH + 'vectors1.txt')
DEVICE_FOUND_YAW = np.genfromtxt(PATH + 'yaw.txt')

SWAP_MATRIX = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])


INTRINSICS1 = np.genfromtxt(PATH + 'align-intrinsics.txt', delimiter=',')
INTRINSICS1 = np.array([[INTRINSICS1[0], 0, INTRINSICS1[2]],
                        [0, INTRINSICS1[1], INTRINSICS1[3]],
                        [0, 0, 1]])

INTRINSICS1 = SWAP_MATRIX.dot(INTRINSICS1).dot(SWAP_MATRIX)
INTRINSICS1[0, 2] = IMAGE1.shape[1] - INTRINSICS1[0, 2]

print('INTRINSICS1:')
print(INTRINSICS1)

INTRINSICS2 = np.genfromtxt(PATH + 'camera-intrinsics.txt', delimiter=',')
INTRINSICS2 = SWAP_MATRIX.dot(INTRINSICS2).dot(SWAP_MATRIX)
INTRINSICS2[0, 2] = IMAGE2.shape[1] - INTRINSICS2[0, 2]
print('INTRINSICS2:')
print(INTRINSICS2)

POSE1 = np.genfromtxt(PATH + 'align-pose.txt', delimiter=',')
print()
print('POSE1')
print(POSE1)

POSE2 = np.genfromtxt(PATH + 'camera-pose.txt', delimiter=',')
print('POSE2')
print(POSE2)

CAMERA_TO_PHONE = R.from_dcm([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

PHONE_TO_GLOBAL1 = R.from_dcm(POSE1[:3, :3])
PHONE_TO_GLOBAL2 = R.from_dcm(POSE2[:3, :3])

POLAR_ANGLE1 = np.arccos(-POSE1[1, 0])
POLAR_ANGLE2 = np.arccos(-POSE2[1, 0])

print()
print('POLAR_ANGLE1:')
print(POLAR_ANGLE1)
print('POLAR_ANGLE2:')
print(POLAR_ANGLE2)


ROTATION_AXIS_GLOBAL1 = np.cross([0, 1, 0], -POSE1[:3, 0])
ROTATION_AXIS_GLOBAL2 = np.cross([0, 1, 0], -POSE2[:3, 0])
ROTATION_AXIS_GLOBAL1 = ROTATION_AXIS_GLOBAL1 / \
    np.linalg.norm(ROTATION_AXIS_GLOBAL1)
ROTATION_AXIS_GLOBAL2 = ROTATION_AXIS_GLOBAL2 / \
    np.linalg.norm(ROTATION_AXIS_GLOBAL2)

print()
print('ROTATION_AXIS_GLOBAL1:')
print(ROTATION_AXIS_GLOBAL1)
print('ROTATION_AXIS_GLOBAL2:')
print(ROTATION_AXIS_GLOBAL2)

GLOBAL_TO_CAMERA1 = CAMERA_TO_PHONE * PHONE_TO_GLOBAL1.inv()
GLOBAL_TO_CAMERA2 = CAMERA_TO_PHONE * PHONE_TO_GLOBAL2.inv()

ROTATION_AXIS_CAMERA1 = GLOBAL_TO_CAMERA1.apply(ROTATION_AXIS_GLOBAL1)
ROTATION_AXIS_CAMERA2 = GLOBAL_TO_CAMERA2.apply(ROTATION_AXIS_GLOBAL2)

print()
print('ROTATION_AXIS_CAMERA1:')
print(ROTATION_AXIS_CAMERA1)
print('ROTATION_AXIS_CAMERA2:')
print(ROTATION_AXIS_CAMERA2)

ROTATION_CAMERA1 = R.from_rotvec(ROTATION_AXIS_CAMERA1 * POLAR_ANGLE1)
ROTATION_CAMERA2 = R.from_rotvec(ROTATION_AXIS_CAMERA2 * POLAR_ANGLE2)

HOMOGRAPHY1 = INTRINSICS1.dot(ROTATION_CAMERA1.as_dcm()).dot(
    np.linalg.inv(INTRINSICS2))
HOMOGRAPHY2 = INTRINSICS2.dot(ROTATION_CAMERA2.as_dcm()).dot(
    np.linalg.inv(INTRINSICS2))

LEVELED_IMAGE1 = cv2.warpPerspective(
    IMAGE1, HOMOGRAPHY1, (IMAGE1.shape[1], IMAGE1.shape[0]))
LEVELED_IMAGE2 = cv2.warpPerspective(
    IMAGE2, HOMOGRAPHY2, (IMAGE2.shape[1], IMAGE2.shape[0]))

cv2.imwrite('level1.png', LEVELED_IMAGE1)
cv2.imwrite('level2.png', LEVELED_IMAGE2)

FEATURE_DESCRIPTOR = cv2.AKAZE_create()

KEYPOINTS1, DESCRIPTORS1 = FEATURE_DESCRIPTOR.detectAndCompute(
    LEVELED_IMAGE1, None)
KEYPOINTS2, DESCRIPTORS2 = FEATURE_DESCRIPTOR.detectAndCompute(
    LEVELED_IMAGE2, None)

MATCHER = cv2.BFMatcher()
MATCHES = MATCHER.knnMatch(DESCRIPTORS1, DESCRIPTORS2, 2)

GOOD_MATCHES = []

for m, n in MATCHES:
    if m.distance < 0.6 * n.distance:
        GOOD_MATCHES.append(m)

CORRESPONDENCES = cv2.drawMatches(
    LEVELED_IMAGE1, KEYPOINTS1, LEVELED_IMAGE2, KEYPOINTS2, GOOD_MATCHES, None)

ORDERED_VECTORS1 = []
ORDERED_VECTORS2 = []

for match in GOOD_MATCHES:
    keypoint1 = KEYPOINTS1[match.queryIdx].pt
    keypoint2 = KEYPOINTS2[match.trainIdx].pt
    # ORDERED_VECTORS1.append(keypoint1)
    # ORDERED_VECTORS2.append(keypoint2)
    ORDERED_VECTORS1.append(np.linalg.inv(
        INTRINSICS1).dot([keypoint1[0], keypoint1[1], 1]))
    ORDERED_VECTORS2.append(np.linalg.inv(
        INTRINSICS2).dot([keypoint2[0], keypoint2[1], 1]))

ORDERED_VECTORS1 = np.array(ORDERED_VECTORS1)
ORDERED_VECTORS2 = np.array(ORDERED_VECTORS2)

print()
print('Ordered Vectors 1:')
print(ORDERED_VECTORS1)
print('Ordered Vectors 2:')
print(ORDERED_VECTORS2)

print('Phone Ordered Vectors 1:')
print(TEST_VECTORS1)
print('Phone Ordered Vectors 2:')
print(TEST_VECTORS2)

ESSENTIAL, _ = cv2.findEssentialMat(
    ORDERED_VECTORS1[:, :2], ORDERED_VECTORS2[:, :2])
NUM_INLIERS, ROTATION, TRANSLATION, _ = cv2.recoverPose(
    ESSENTIAL, ORDERED_VECTORS1[:, :2], ORDERED_VECTORS2[:, :2])

TEST_ESSENTIAL, _ = cv2.findEssentialMat(TEST_VECTORS1, TEST_VECTORS2)
TEST_NUM_INLIERS, TEST_ROTATION, TEST_TRANSLATION, _ = cv2.recoverPose(
    TEST_ESSENTIAL, TEST_VECTORS1, TEST_VECTORS2)

SAMPLE_ESSENTIAL = np.genfromtxt(PATH + 'essential.txt')
SAMPLE_NUM_INLIERS, SAMPLE_ROTATION, SAMPLE_TRANSLATION, _ = cv2.recoverPose(
    SAMPLE_ESSENTIAL, TEST_VECTORS1, TEST_VECTORS2)

ROTATED_Z = ROTATION.dot([0, 0, 1])
YAW = np.arctan2(*ROTATED_Z[[0, 2]])

TEST_ROTATED_Z = TEST_ROTATION.dot([0, 0, 1])
TEST_YAW = np.arctan2(*TEST_ROTATED_Z[[0, 2]])

print()
print('NUM_INLIERS:')
print(NUM_INLIERS)
print('ESSENTIAL:')
print(ESSENTIAL)
print('ROTATION:')
print(ROTATION)
print('TRANSLATION:')
print(TRANSLATION)
print('YAW (degrees):')
print(YAW * 180 / np.pi)
print('Test YAW (degrees)')
print(TEST_YAW * 180 / np.pi)
print('Test ROTATION:')
print(TEST_ROTATION)
print('Test ESSENTIAL:')
print(TEST_ESSENTIAL)
print('Sample ROTATION:')
print(SAMPLE_ROTATION)

cv2.imshow('CORRESPONDENCES', CORRESPONDENCES)
while (cv2.waitKey() != ord('q')):
    pass
cv2.destroyAllWindows()

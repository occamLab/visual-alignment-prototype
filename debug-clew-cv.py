#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# path='AppData/Documents/2019-07-25-15-25-52/'
path='AppData/Documents/2019-07-26-15-56-44'
print(path + 'align-image.png')
image1 = cv2.imread(path + 'align-image.png', 0)
image2 = cv2.imread(path + 'camera-image.png', 0)

image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)


intrinsics1 = np.array([[1558.10254, 0, 719.077759],
                        [0, 1558.10254, 933.125976],
                        [0, 0, 1]])

intrinsics2 = intrinsics1

# pose1 = np.array([[0.136009559, -0.988509237, 0.0659611151, 0],
#                   [0.545460939, 0.0191378202, -0.837917745, 0],
#                   [0.827027082, 0.149944037, 0.541796088, 0],
#                   [0.762800991, -0.0225746147, -0.982956528, 0.99999988]]).T

# pose2 = np.array([[0.609848022, - 0.773755192, 0.17143032, 0],
#                   [0.780520796, 0.623892009, 0.0393198319, 0],
#                   [-0.137377933, 0.109825782, 0.984411239, 0],
#                   [0.00732497917, 0.0535333417, 0.0273590703, 0.99999994]]).T

pose1 = np.array(((-0.227074564, -0.925714254, -0.302473277, 0),
                  (0.873773694, -0.0565025322, -0.483039618, 0),
                  (0.430066198, -0.373979062, 0.821695268, 0),
                  (-0.107774988, 0.0566815138, 0.0857271701, 1.00000012))).T

pose2 = np.array(((0.0855018944, -0.662365019, 0.74428618, 0),
         (-0.742346584, -0.5406003, -0.395819247, 0),
         (0.664538026, -0.51867497, -0.537926674, 0),
         (-0.260924339, 0.360042274, 0.354705989, 0.99999988))).T

phone_to_camera = R.from_dcm([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
global_to_phone1 = R.from_dcm(pose1[:3, :3])
global_to_phone2 = R.from_dcm(pose2[:3, :3])

polar_angle1 = np.arccos(-pose1[1, 0])
polar_angle2 = np.arccos(-pose2[1, 0])
rotation_axis1 = np.cross([0, 1, 0], -pose1[:3, 0])
rotation_axis2 = np.cross([0, 1, 0], -pose2[:3, 0])
rotation_axis1 = rotation_axis1 / np.linalg.norm(rotation_axis1)
rotation_axis2 = rotation_axis2 / np.linalg.norm(rotation_axis2)

square_rotation_global1 = R.from_rotvec(rotation_axis1 * polar_angle1)
square_rotation_global2 = R.from_rotvec(rotation_axis2 * polar_angle2)

# global_to_camera1 = global_to_phone1 * phone_to_camera
# global_to_camera2 = global_to_phone2 * phone_to_camera
global_to_camera1 = phone_to_camera * global_to_phone1.inv()
global_to_camera2 = phone_to_camera * global_to_phone2.inv()

axis_camera1 = global_to_camera1.apply(rotation_axis1)
axis_camera2 = global_to_camera2.apply(rotation_axis2)
test_rotation1 = R.from_rotvec(axis_camera1 * polar_angle1)
test_rotation2 = R.from_rotvec(axis_camera2 * polar_angle2)

# homography1 = intrinsics1.dot(a1.as_dcm()).dot(np.linalg.inv(intrinsics1))
# homography2 = intrinsics2.dot(a2.as_dcm()).dot(np.linalg.inv(intrinsics2))
homography1 = intrinsics1.dot(test_rotation1.as_dcm()).dot(
    np.linalg.inv(intrinsics1))
homography2 = intrinsics2.dot(test_rotation2.as_dcm()).dot(
    np.linalg.inv(intrinsics2))

squared_image1 = cv2.warpPerspective(
    image1, homography1, (image1.shape[1], image1.shape[0]))
squared_image2 = cv2.warpPerspective(
    image2, homography2, (image2.shape[1], image2.shape[0]))

feature_descriptor = cv2.AKAZE_create()

keypoints1, descriptors1 = feature_descriptor.detectAndCompute(
    squared_image1, None)
keypoints2, descriptors2 = feature_descriptor.detectAndCompute(
    squared_image2, None)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, 2)

good_matches = []

# Lowe ratio test

for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)

ordered_vectors1 = []
ordered_vectors2 = []

# Order the points and vectors to be matched by index.
for match in good_matches:
    keypoint1 = keypoints1[match.queryIdx].pt
    keypoint2 = keypoints2[match.trainIdx].pt
    ordered_vectors1.append(np.linalg.inv(
        intrinsics1).dot([keypoint1[0], keypoint1[1], 1]))
    ordered_vectors2.append(np.linalg.inv(
        intrinsics2).dot([keypoint2[0], keypoint2[1], 1]))

# ordered_vectors1 = np.array(ordered_vectors1)[:, :2]
# ordered_vectors2 = np.array(ordered_vectors2)[:, :2]

correspondances = cv2.drawMatches(
    squared_image1, keypoints1, squared_image2, keypoints2, good_matches, None)


# essential_mat, _ = cv2.findEssentialMat(ordered_vectors1, ordered_vectors2)
# num_inliers, rotation_mat, translation, _ = cv2.recoverPose(
#     essential_mat, ordered_vectors1, ordered_vectors2)

# print(rotation_mat)
# print(R.from_dcm(rotation_mat).as_euler('yxz', True))

cv2.imshow('debug', correspondances)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
 
 
def get_normalized_vector(image_coordinate, focal_length, ppx, ppy):
    '''Convert an image coordinate to a vector with z=1.
 
    Args:
        image_coordinate: The image coordinate in (column, row) format.
        focal_length: The focal length of the camera.
        ppx: The x coordinate of the optical center of the camera.
        ppy: The y coordinate of the optical center of the camera.
 
    Returns:
        The x and y components of the vector representing the ray from
        the camera to the pixel, assuming the z component is 1.
    '''
    new_coordinate = np.array(
        [image_coordinate[0] - ppx, image_coordinate[1] - ppy])
    return new_coordinate / focal_length
 
image1 = cv2.imread('level1.png', 0)
image2 = cv2.imread('level2.png', 0)
 
feature_descriptor = cv2.AKAZE_create()
 
keypoints1, descriptors1 = feature_descriptor.detectAndCompute(
    image1, None)
keypoints2, descriptors2 = feature_descriptor.detectAndCompute(
    image2, None)
 
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, 2)
 
good_matches = []
 
# Lowe ratio test
 
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)
 
intrinsics1 = np.array([[1.559217e+03, 0.000000e+00, 7.183415e+02],
                        [0.000000e+00, 1.559217e+03, 9.334793e+02],
                        [0.000000e+00, 0.000000e+00, 1.000000e+00]])
 
intrinsics2 = np.array([[1558.10254, 0, 719.077759],
                        [0, 1558.10254, 933.125976],
                        [0, 0, 1]])


correspondences = cv2.drawMatches(
    image1, keypoints1, image2, keypoints2, good_matches, None)
 
ordered_vectors1 = []
ordered_vectors2 = []
 
# Order the points and vectors to be matched by index.
for match in matches:
    keypoint1 = keypoints1[match[0].queryIdx].pt
    keypoint2 = keypoints2[match[0].trainIdx].pt
    ordered_vectors1.append(get_normalized_vector(
        keypoint1, intrinsics1[0, 0], intrinsics1[0, 2], intrinsics1[1, 2]))
    ordered_vectors2.append(get_normalized_vector(
        keypoint2, intrinsics2[0, 0], intrinsics2[0, 2], intrinsics2[1, 2]))
 
ordered_vectors1 = np.array(ordered_vectors1)
ordered_vectors2 = np.array(ordered_vectors2)
 
 
essential_mat, _ = cv2.findEssentialMat(ordered_vectors1, ordered_vectors2)
num_inliers, rotation_mat, translation, _ = cv2.recoverPose(
    essential_mat, ordered_vectors1, ordered_vectors2)
 
rotation = R.from_dcm(rotation_mat)
ypr = rotation.as_euler('yxz', degrees=True)
print(ypr[0])

ROTATED_Z = rotation.as_dcm().dot([0, 0, 1])
YAW = np.arctan2(*ROTATED_Z[[0, 2]])


print(YAW)
cv2.imshow('CORRESPONDENCES', correspondences)
while (cv2.waitKey() != ord('q')):
    pass
cv2.destroyAllWindows()

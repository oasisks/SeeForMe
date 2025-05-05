import numpy as np

import cv2

#The following collection of pixel locations and corresponding relative
#ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_IMAGE_PLANE = [[1315, 1065],
                   [480, 985],
                   [840, 945],
                   [1550, 915],
                   [1175, 870],
                   [1285, 790],
                   [1005, 770]]
######################################################
# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
X_OFFSET = 2.5
PTS_GROUND_PLANE = [[63+X_OFFSET, -11.8],
                    [69.5+X_OFFSET, 30],
                    [76+X_OFFSET, 11.6],
                    [84.125+X_OFFSET, -30],
                    [90.5+X_OFFSET, -8.27],
                    [108.5+X_OFFSET, -18],
                    [111.75+X_OFFSET, 3.35]]
######################################################

# METERS_PER_INCH = 0.0254
# CONVERSION = 0.01

class Homog():
    def __init__(self):
        #Initialize data into a homography matrix

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * 1.0
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in inches.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y
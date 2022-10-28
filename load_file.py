from scipy.io import loadmat
# import pandas as pd
import numpy as np
import math
# import cv2

# loading data of pushing points from text file into 2D array
pushing_points = [[0 for i in range(4)] for j in range(9)]
with open("trial_1/pushing_points.txt") as f:
    i = 0
    for line in f:        
        points = line[line.find('[') + 1: line.find(']')]
        point = points.split()
        
        pushing_points[i][0] = point[0]
        pushing_points[i][1] = point[1]
        pushing_points[i][2] = point[2]
        pushing_points[i][3] = 1
        i += 1
f.close()
print("Pushing-Points ")
print(pushing_points[0])

# loading projection matrix from data for camera to base 
annots = loadmat('trial_1/trans_data.mat')
matrix_cTb = [[element for element in upperElement] for upperElement in annots['trans']]
# print("Intrinsic Matrix")
# print(matrix_cTb)

# inverting matrix from base to camera
matrix_bTc = np.linalg.inv(matrix_cTb)

pushing_points = np.array(pushing_points, dtype=float)
camera_coords = np.dot(matrix_bTc, pushing_points[0])

# printing result matrix (of points in )
""" columns = ['x', 'y', 'z' '1']
df = pd.DataFrame (camera_coords, columns=columns)
print(df) """

# camera coordinates to pixel coordinates
img_width = 640
img_height = 480
fov = 45
near = 0.01
aspect_ratio = img_width / img_height
e = 1 / (np.tan(np.radians(fov/2.)))
t = near / e; b = -t
r = t * aspect_ratio; l = -r
alpha = img_width / (r-l) # pixels per meter
focal_length = near * alpha 
fx = focal_length; fy = focal_length

# x, y, z in camera coordinates
print("Camera Coordinates")
print(camera_coords)

Xc = camera_coords[0]
Yc = camera_coords[1]
Zc = camera_coords[2]

Px = img_width / 2 # principal off set  
Py = img_height / 2


# calculating x , y in pixel coordinates
Xp = (Xc / Zc) + Px
Yp = (Yc / Zc) + Py

print(str(Xp) + ", " + str(Yp))

# Mapping point on Image using OpenCV
""" img = cv2.imread('trial_36/rgb1.png',cv2.IMREAD_COLOR)
cv2.circle(img,(Xp, Yp), 5, (0,0,255), -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows() """
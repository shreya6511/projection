from scipy.io import loadmat
import pandas as pd
import numpy as np
import math
import cv2

# loading data of pushing points from text file into 2D array
pushing_points = [[0 for i in range(4)] for j in range(9)]
with open("trial_37/pushing_history.txt") as f:
    i = 0
    for line in f:
        
        direction = line[line.find(':') + 1: line.find(',')]
        if(direction == 'right'):
            pushing_points[i][0] = 1
        else:
            pushing_points[i][0] = -1
        
        points = line[line.find('[') + 1: line.find(']')]
        point = points.split()
        
        pushing_points[i][1] = point[0]
        pushing_points[i][2] = point[1]
        pushing_points[i][3] = point[2]
        i += 1
f.close()

# loading projection matrix from data for camera to base 
annots = loadmat('trans_data_35.mat')
proj_matrix_cTb = [[element for element in upperElement] for upperElement in annots['trans']]

# inverting matrix to be the projection matrix from base to camera
proj_matrix_bTc= np.linalg.inv(proj_matrix_cTb)
print(proj_matrix_bTc)
pushing_points = np.array(pushing_points, dtype=float)
camera_coords = np.dot(pushing_points, proj_matrix_bTc)

# printing result matrix (of points in )
columns = ['dir', 'x', 'y', 'z']
df = pd.DataFrame (camera_coords, columns=columns)
print(df)

# camera coordinates to pixel coordinates
img_width = 480
img_height = 640
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
Xc = camera_coords[1][1]
Yc = camera_coords[1][2]
Zc = camera_coords[1][3]

Px = img_height / 2 # principal off set  
Py = img_width / 2

# calculating x , y in pixel coordinates
Xp = fx * (Xc / Zc) + Px
Yp = fy * (Yc / Zc) + Py

print(Xp)
print(Yp)


"""# Mapping point on Image using OpenCV
img = cv2.imread('trial_37/rgb1.png',cv2.IMREAD_COLOR)
cv2.circle(img,(25,25), 5, (0,0,255), -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
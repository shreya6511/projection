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

proj_matrix_cTb = np.array(proj_matrix_cTb, dtype= float)
pushing_points = np.array(pushing_points, dtype=float)
""" print(str(len(pushing_points)) + " x " + str(len(pushing_points[0])))
print(str(len(proj_matrix_bTc)) + " x " + str(len(proj_matrix_bTc[0]))) """
camera_coords = np.dot(pushing_points, proj_matrix_bTc)

# printing result matrix (of points in )
columns = ['dir', 'x', 'y', 'z']
df = pd.DataFrame (camera_coords, columns=columns)
print(df)

# camera coordinates to pixel coordinates
p_u = 1 # how many meters wide a pixel is
p_v = 1
c_x = 1 # origin of image coordinates at center of image plane translation to pixel coordinate center at top left
c_y = 1
trans_matrix = np.matrix([[1/p_u, 0, c_x],
                [0, 1/p_v, c_y], 
                [0, 0, 1]])

# Mapping point on Image
img = cv2.imread('trial_37/rgb1.png',cv2.IMREAD_COLOR)
cv2.circle(img,(25,25), 5, (0,0,255), -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" annots = loadmat('trans_data_35.mat')
#print(annots)

# printing as a list
con_list = [[element for element in upperElement] for upperElement in annots['trans']]
#print(con_list)
data = np.array(con_list[0])

# using pandas to display data in tabular form
print("Transformation Data in 3D Base Coordinates")
columns = ['x', 'y', 'z', 'dir']
df = pd.DataFrame (con_list, columns=columns)
print(df) """

""" # creating the projection matrix 
focal_len = 320 # calculated using width of image and hfov
px = 0 # principal offset in x direction
py = 0 # principal offset in y direction
# camera instrincs: focal length, axis skew, 
K = np.matrix([[focal_len, 0, 0, 0], 
              [0, focal_len, 0, 0],
              [0, 0, 1, 0]])

# camera extrinsics: assuming camera and base share same coordinate system

# rotation matrix
theta = 90 # angle of rotation
Rx = np.matrix([[1, 0 , 0, 0], 
                [0, math.cos(theta), (math.sin(theta) * -1), 0], 
                [0, math.sin(theta), math.cos(theta), 0], 
                [0, 0, 0, 1]])
Ry = np.matrix([[math.cos(theta), 0 , math.sin(theta), 0], 
                [0, 1, 0, 0], 
                [(math.sin(theta) * -1), 0, math.cos(theta), 0], 
                [0, 0, 0, 1]])
Rz = np.matrix([[math.cos(theta), (math.sin(theta) * -1) , 0, 0], 
                [math.sin(theta), math.cos(theta), 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]])

# translation matrix
tx = 1
ty = 1
tz = 1
Txyz = np.matrix([[0, 0, 0, tx],
                [0, 0, 0, ty], 
                [0, 0, 0, tz], 
                [0, 0, 0, 1]])

camera_extrin = np.dot((np.dot(np.dot(Rx, Ry), Rz)), Txyz)
print(camera_extrin)

projection_mat = np.dot(K, camera_extrin)

camera_coords = np.dot(data, projection_mat) """

""" # Make empty black image
image=np.zeros((20,40,3),np.uint8)
# Make one pixel red
image[10,5]=[0,0,255]
# Save
cv2.imwrite("result.png",image) """

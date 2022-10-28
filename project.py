from scipy.io import loadmat
import numpy as np

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
print("Pushing-Points")
print(pushing_points[0])

# calculating values for intrinsic matrix
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

Px = img_width / 2 # principal off set  
Py = img_height / 2

axis_skew = -1

intrinsic = np.array([[fx, axis_skew, Px], [0, fy, Py], [0, 0, 1]])
print(intrinsic)
print(intrinsic.shape)

annots = loadmat('trial_1/trans_data.mat')
trans_mat = [[element for element in upperElement] for upperElement in annots['trans']] # maybe just the 3D Rotation
extrinsic = np.asarray(trans_mat)
print(extrinsic.shape)
projection = np.dot(intrinsic, extrinsic)

camera_coords = np.dot(projection, pushing_points[0])

print(camera_coords)


import cv2

# loading data of pushing points from text file into 2D array
pushing_points = [[0 for i in range(3)] for j in range(9)]
with open("trial_1/pushing_history.txt") as f:
    i = 0
    for line in f:        
        points = line[line.find('[') + 1: line.find(']')]
        point = points.split()
        
        pushing_points[i][0] = point[0][:-1]
        pushing_points[i][1] = point[1][:-1]
        pushing_points[i][2] = point[2][:-1]

        i += 1
        
f.close()

img_width = 640
img_height = 480

Px = img_width / 2 # principal off set  
Py = img_height / 2

for k in range(9):
    camera_coords = pushing_points[k]
    
    Xc = float(camera_coords[0])
    Yc = float(camera_coords[1])
    Zc = float(camera_coords[2])
    
    # calculating x , y in pixel coordinates
    Xp = (Xc / Zc) + float(Px)
    Yp = (Yc / Zc) + float(Py)
    
    # Mapping point on Image using OpenCV
    fileName = "trial_1/rgb" + str(k+1) + ".png"
    # fileName = "trial_1/rgb2.png"
    print(fileName)
    
    img = cv2.imread(fileName,cv2.IMREAD_COLOR)
    cv2.circle(img,(int(Xp), int(Yp)), 5, (0,0,255), -1)
    cv2.imwrite(fileName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    k += 1

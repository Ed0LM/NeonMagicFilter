import cv2 
import numpy as np

def ScharrEdge(image):
    image = cv2.GaussianBlur(image, (9,9), 0)
    b,g,r = cv2.split(image)
    b_x = cv2.Scharr(b, cv2.CV_64F, 1, 0)  # X direction
    b_y = cv2.Scharr(b, cv2.CV_64F, 0, 1)  # Y direction

    g_x = cv2.Scharr(g, cv2.CV_64F, 1, 0)  # X direction
    g_y = cv2.Scharr(g, cv2.CV_64F, 0, 1)  # Y direction

    r_x = cv2.Scharr(r, cv2.CV_64F, 1, 0)  # X direction
    r_y = cv2.Scharr(r, cv2.CV_64F, 0, 1)  # Y direction

    magn_b = np.sqrt(b_x**2 + b_y**2)
    magn_g = np.sqrt(g_x**2 + g_y**2)
    magn_r = np.sqrt(r_x**2 + r_y**2)

    magn_b = cv2.convertScaleAbs(magn_b)
    magn_g = cv2.convertScaleAbs(magn_g)
    magn_r = cv2.convertScaleAbs(magn_r)

    merged = cv2.merge((magn_b,magn_g,magn_r))
    
    return merged
   
def DarkFilter(image):

    threshold = 20
    dark_mask = np.all(image < threshold, axis=2)
    image[dark_mask] = [0, 0, 0]
    return image



input_path = "input_path"
output_path = "path"
output_path1 = "path"


img = cv2.imread(input_path) 
elab = ScharrEdge(img)
elab = DarkFilter(elab)
status = cv2.imwrite(output_path,elab)

image_32f = elab.astype(np.float32)
gray_image = cv2.cvtColor(image_32f, cv2.COLOR_BGR2GRAY)          #to obtain grayscale version of it
cv2.imwrite(output_path1,gray_image)

print("Image written to file-system : ",status)
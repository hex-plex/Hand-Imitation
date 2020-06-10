import numpy as np
import cv2
# load the image
image = cv2.imread("webcam_1.jpg")
# skin color boundaries
lower = np.array([100,100,100])
upper = np.array([255,255,255])
#mask in the mentioned range
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)
ret,thresh = cv2.threshold(mask, 40, 255, 0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy=hierarchy[0]
if len(contours) != 0:
    for c in zip(contours, hierarchy):
         currentContour = c[0]
         currentHierarchy = c[1]
         x,y,w,h = cv2.boundingRect(currentContour)
         if currentHierarchy[3] < 0:
            # these are the outermost parent components
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.drawContours(output, currentContour, -1, 130, 3)
            crop_img = output[y:y+h, x:x+w]


# show the images
cv2.imshow("Cropped",crop_img)
cv2.imshow("im",image)
cv2.waitKey(0)

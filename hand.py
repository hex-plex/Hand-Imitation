import cv2
import numpy as np

hsv = cv2.imread("handt.jpg")
#r=cv2.selectROI(hsv)
img = cv2.cvtColor(hsv,cv2.COLOR_BGR2LAB)
#imp=img[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
#low=np.array([imp[:,:,0].min(),imp[:,:,1].min(),imp[:,:,2].min()])
#high=np.array([imp[:,:,0].max(),imp[:,:,1].max(),imp[:,:,2].max()])
low = np.array([0, 130, 117])
high = np.array([250, 140, 135])
print(low)
print(high)
mask = cv2.inRange(img,low,high)
kernel = np.array([3,3],dtype=np.uint8)
mask= cv2.dilate(mask,kernel,iterations = 3)
cv2.imshow("all the points",mask)
cv2.waitKey(0)
contour,heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contheir=zip(contour,np.squeeze(heirarchy))
sortList = sorted(contheir,key=lambda x: cv2.contourArea(x[0]),reverse=True)
temp=[]
convexs = []
convexsp=[]
some_details=[]
epsilon=0.7*cv2.contourArea(sortList[0][0])
for cnt,heir in sortList:
    if cv2.contourArea(cnt)< epsilon: break
    if heir[-1]!=-1:
        continue
    else:
        temp.append(cnt)
        convexs.append(cv2.convexHull(cnt,returnPoints=False))
        convexsp.append(cv2.convexHull(cnt))
        some_details.append(cv2.approxPolyDP(cnt,0.004*cv2.arcLength(cnt,True),True))

cv2.drawContours(hsv,temp,-1,(255,0,0),2)
cv2.imshow("al",hsv)
cv2.waitKey(0)
cv2.drawContours(hsv,convexsp,-1,(0,255,0),2)
cv2.imshow("al",hsv)
cv2.waitKey(0)
cv2.drawContours(hsv,some_details,-1,(0,0,255),2)
cv2.imshow("al",hsv)
cv2.waitKey(0)

for i in range(len(convexs)):
    print(cv2.convexityDefects(temp[i],convexs[i]))
for i in range(len(convexs)):
    print(len(some_details[i]))
    if len(some_details[i])<34:
        cv2.drawContours(hsv,[temp[i]],-1,(255,225,0),2)
        cv2.imshow("al",hsv)
        cv2.waitKey(0)

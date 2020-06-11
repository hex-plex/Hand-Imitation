def handmask(frame):
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)
    #define region of interest
    roi=frame[100:400, 100:300]
    cv2.rectangle(frame,(100,100),(300,400),(0,255,0),0)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    # define range of skin color in lab
    lower_skin = np.array([40,120,120], dtype=np.uint8)
    upper_skin = np.array([255,170,170], dtype=np.uint8)
    #extract skin colour image
    mask = cv2.inRange(lab, lower_skin, upper_skin)
    #blur the image
    cr_frame=frame[100:400,100:300]
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    cv2.imshow("cropped",cr_frame)
    return mask

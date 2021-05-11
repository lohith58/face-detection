import cv2

detect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imp_img= cv2.VideoCapture("sundar.jpg")

res,img =imp_img.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#To detects faces of different sizes from the input image optained
faces =detect.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
        # To draw a rectangle for the face in image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow('detecting faces',img)
k = cv2.waitKey(0)

# Close the window
imp_img.release()

# De-allocate any associated memory usage that we used as part of our program
cv2.destroyAllWindows()

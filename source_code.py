'''
team name: 꾸짖을 갈!
team member: 202235009 권민준, 202235021 김세린, 202235061 송재곤, 202235119 정승호
function: Mosaic after face recognition, count detected numbers
'''

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face.jpg') # Import Image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Switch images to black and white
faces = face_cascade.detectMultiScale(gray, 1.3, 4) # Face detection

i = 1 # count start number
v = 20

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 190), 5) # Drawing a box
    cv2.putText(img, str(i), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1) # output count numbers to the screen

    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]

    roi = cv2.resize(roi_color, (w // v, h // v))
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
    img[y:y+h, x:x+w] = roi

    i = i + 1 # number count

# Display the total number counted
cv2.putText(img, 'Found number: ', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(img, str(faces.shape[0]), (280, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

# Image Detection
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

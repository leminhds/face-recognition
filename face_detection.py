import cv2
import os

# get pretrain cascade classifier

cascades_cf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img, draw=True):
    # convert image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    faces = cascades_cf.detectMultiScale(grayscale_img, 
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    face_box, face_coords = None, []
    
    # draw box aroudn detected faces
    for (x, y, width, height) in faces:
        if draw:
            cv2.rectangle(img, (x, y), (x+ width, y+height), (0, 255, 0), 5)
        face_box = img[y:y+height, x:x+width]
        face_coords = [x, y, width, height]
    return img, face_box, face_coords
    

files = os.listdir('sample_faces')
images = [file for file in files if 'jpg' in file]
for image in images:
    img = cv2.imread('sample_faces/' + image)
    detected_faces, _, _ = detect_faces(img)
    cv2.imwrite('sample_faces/box_drawn/' + image, detected_faces)
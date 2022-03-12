import cv2
import math

import utils
import face_detection

counter = 5

video_capture = cv2.VideoCapture(0)

# give user 5 second before photo is taken
while True:
    _, frame = video_capture.read()
    frame, face_box, face_coords = face_detection, face_detection.detect_faces(frame)
    text = 'Image will be taken in {}..'.format(math.ceil(counter))
    if face_box is not None:
        frame = utils.write_on_frame(frame, text, face_coords[0], face_coords[1] - 10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if counter <= 0:
        cv2.imwrite('true_img.png', face_box)
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Onboarding Image Captured")

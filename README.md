# face-recognition
Face recognition using Siamese model 


The problem can be broken down into:
- face detection: detect and isolate face in the image. if there are multiple faces, we need to detect each of them separately
- face recognition: each detected face is run through a neural network to classify the subject

Face detection used here is Haar Cascades (using cv2)
the idea behind Haar Cascades is that all human faces share certain properties, such as:
- area of the eye is darker than the forehead and the cheeks
- the area of the nose is brighter

Face recognition use Siamese model. The steps are:
- use convolutional layers to extract identifying features from faces --> map to a lower dimension (128 x 1) vector
- using Euclidean distance to measure differences



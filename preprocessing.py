import dlib
import cv2
import numpy as np

def HorizontalFlip(batch_img, p=0.5):
    # (T, H, W, C)
    if np.random.random() > p:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

class LipDetector:
  '''
    A class to find lip in a face image
  '''
  def __init__(self):
    # Load the face detector and shape predictor
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


  def findlip(self, im, extra = 10):
    # Detect faces in the grayscale image
    grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    faces = self.detector(im)
    if faces:
        # Assuming there's only one face detected
        face = faces[0]

        # Detect facial landmarks
        landmarks = self.predictor(grey, face)

        # Extract the lip region using landmarks
        lip_x = landmarks.part(48).x  # Left corner of the mouth
        lip_y = landmarks.part(51).y  # Top of the upper lip
        lip_w = landmarks.part(54).x - landmarks.part(48).x  # Width of the mouth
        lip_h = landmarks.part(57).y - landmarks.part(51).y  # Height of the upper lip

        # Extract the lip region
        lip_region = im[lip_y-extra:lip_y+extra + lip_h, lip_x-extra:lip_x+extra + lip_w]
        return lip_region
    
    else:
        print("No face detected in the image.")

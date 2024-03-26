from instagram_private_api import Client, ClientCompatPatch
import urllib.request
import dlib
import cv2
import numpy as np
import os
from joblib import load
# Replace with your own credentials
username = 'rakesh091196'
password = 

api = Client(username, password)
user_id = input("Enter a insta user ID: ")
profile = api.username_info(user_id)
#print(profile)
profile_picture_url = profile['user']['hd_profile_pic_url_info']['url']

# Download the profile picture
# response = api.client_session.get(profile_picture_url)
# with open('profile_picture.jpg', 'wb') as f:
    # f.write(response.content)
urllib.request.urlretrieve(profile_picture_url, "profile_picture.jpg")

def warp_face(image, landmarks):
  """Warps the face in the given image to the mean face.

  Args:
    image: The image to warp.
    landmarks: The landmarks of the face in the image.

  Returns:
    The warped image.
  """

  # Find the mean face.
  mean_face = np.mean(landmarks, axis=0)

  # Compute the difference between the mean face and the landmarks of the current face.
  difference = landmarks - mean_face

  # Use a least squares fit to find a transformation that maps the landmarks of the current face to the mean face.
  transformation = cv2.estimateAffine2D(difference, mean_face)

  # Warp the image using the transformation.
  warped_image = cv2.warpAffine(image, transformation, image.shape[:2])

  return warped_image
  
  

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
features = []
# Load the image and convert to grayscale
image = cv2.imread('profile_picture.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

for face in faces:
    # Predict the landmarks for the detected face
    landmarks = predictor(gray, face)
    pts = np.zeros((51, 2))
    for i in range(51):
        pts[i] = (landmarks.part(i+17).x, landmarks.part(i+17).y)
    warped_pts = pts
    # Normalize the warped landmarks to the [0, 1] x [0, 1] region
    warped_pts[:, 0] /= image.shape[1]
    warped_pts[:, 1] /= image.shape[0]
    # Concatenate the x-coordinates and y-coordinates into a single feature vector
    feature_vector = np.concatenate([warped_pts[:, 0], warped_pts[:, 1]])

    # Append the feature vector to the list of features
    features.append(feature_vector)
# Convert the list of features to a numpy array
features = np.array(features)

# load the saved model
svm_model = load('svm_model.joblib')

# make real-time predictions
X_test = features # load the test data
y_pred = svm_model.predict(X_test)

if (y_pred[0] == 1):
    print('the profile photo of the given public account is not GAN generated')
elif (y_pred[0] == 0):
    print('the profile photo of the given public account is GAN generated')
else:
    print('cannot abke to determine')

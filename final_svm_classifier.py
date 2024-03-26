import dlib
import cv2
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

def extract_features(folder_path_file)

# Load the face detector and landmark predictor from dlib
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("D:/Rakesh/cvproj/shape_predictor_68_face_landmarks.dat")

# Define the folder containing the images
	folder_path = folder_path_file  # Replace with your own folder path

# Initialize an empty list to store the feature vectors
	features = []

# Iterate over each image in the folder
	for filename in os.listdir(folder_path):
    # Load the input image
		image = cv2.imread(os.path.join(folder_path, filename))

    # Convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face in the grayscale image
		faces = detector(gray)

    # Iterate over the detected faces
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
	#np.save(dest_file, features)
	return(features)

real_features = extract_features("D:/Rakesh/cvproj/real")
gan_features = extract_features("D:/Rakesh/cvproj/gan_dataset/final")
# Load the feature vectors of GAN and real images and their corresponding labels
#gan_features = np.load('features_gan.npy') #real_features = np.load('features_orig.npy')
gan_labels = np.zeros(len(gan_features))
real_labels = np.ones(len(real_features))
X = np.concatenate((gan_features, real_features), axis=0)
y = np.concatenate((gan_labels, real_labels), axis=0)

# Split the data into train and test sets with 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the SVM classifier
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
}

# Define the 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute the inverse class frequencies for balancing the losses of the two classes
class_weights = len(y_train) / (2 * np.bincount(y_train.astype(int)))

# Define the SVM classifier with RBF kernel and balanced sample weights
svm = SVC(kernel='rbf', class_weight='balanced')

# Define the grid search object with SVM classifier and parameter grid
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)




# Train the SVM classifier with feature vectors and corresponding labels
grid_search.fit(X_train, y_train, sample_weight=class_weights[y_train.astype(int)])

# Print the best hyperparameters and the corresponding accuracy
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy: ", grid_search.best_score_)

dump(grid_search, 'svm_model.joblib')

# Predict the labels of test data using the fitted SVM classifier
y_pred = grid_search.predict(X_test)
print(y_pred)
print(y_test)
# Calculate the accuracy of predicted labels on test data
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy: ", accuracy)
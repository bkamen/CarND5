# routine to create classifier for the project
# not included to main routine, to not have to do it constantly

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from TrackingFunctions import *
import glob
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt


# Divide up into cars and notcars
images = glob.glob('./train_images/vehicles/**/*.png', recursive=True)
cars = []
for image in images:
    cars.append(image)

images = glob.glob('./train_images/non-vehicles/**/*.png', recursive=True)
notcars = []
for image in images:
    notcars.append(image)

#sample_size = 100
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]


# Parameters
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

t = time.time()
# extraction of the features of the car images
car_features_color, car_features_hog = extract_features(cars, color_space, spatial_size, hist_bins, orient,
                                                        pix_per_cell, cell_per_block, hog_channel, spatial_feat,
                                                        hist_feat, hog_feat)
# extraction of the features of the non-car images
notcar_features_color, notcar_features_hog = extract_features(notcars, color_space, spatial_size, hist_bins, orient,
                                                              pix_per_cell, cell_per_block, hog_channel, spatial_feat,
                                                              hist_feat, hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

# scaling color features
X_color = np.concatenate((car_features_color, notcar_features_color))
# Fit a per-column scaler
X_scaler_color = StandardScaler().fit(X_color)
# Apply the scaler to X
scaled_X_color = X_scaler_color.transform(X_color)

# scaling shape features
X_hog = np.concatenate((car_features_hog, notcar_features_hog))
# Fit a per-column scaler
X_scaler_hog = StandardScaler().fit(X_hog)
# Apply the scaler to X
scaled_X_hog = X_scaler_hog.transform(X_hog)

# feature vector plot
plt.figure()
plt.plot(X_color[0])
plt.ylabel('color feature vector')
#plt.show()
plt.savefig('./output_images/color_features.png')

plt.figure()
plt.plot(X_hog[0])
plt.ylabel('HoG feature vector')
#plt.show()
plt.savefig('./output_images/hog_features.png')

plt.figure()
plt.plot(scaled_X_color[0], label='scaled color features')
plt.plot(scaled_X_hog[0], label='scaled HoG features')
plt.legend()
plt.savefig('./output_images/scaled_features.png')

# concatenate features
scaled_X = np.concatenate((scaled_X_color, scaled_X_hog), axis=1)

# Define the labels vector
y = np.hstack((np.ones(len(car_features_color)), np.zeros(len(notcar_features_color))))

# remove features with too low variance
#scaled_X = SelectKBest(f_classif, k=int(.7*len(scaled_X[0]))).fit_transform(scaled_X, y)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
print('Feature vector length:', len(X_train[0]))

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
#svc = LinearSVC()
svc = SVC(kernel='rbf')
#svc = RandomForestClassifier(n_estimators=50, n_jobs=-1)
#svc = AdaBoostClassifier(n_estimators=50)
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

if round(svc.score(X_test, y_test), 4) > .97:
    joblib.dump(svc, 'CarClassifier.pkl')
    joblib.dump(X_scaler_color, 'ScalerColor.pkl')
    joblib.dump(X_scaler_hog, 'ScalerHOG.pkl')

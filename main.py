from TrackingFunctions import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Parameters
y_start_stop = [None, None]  # Min and max in y to search in slide_window()
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# load classifier and scaler
svc = joblib.load('CarClassifier.pkl')
X_scaler = joblib.load('scaler.pkl')

image = mpimg.imread('./test_images/test1.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(128, 128),
                       xy_overlap=(0.7, 0.7))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                             hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.show()

ystart = 300
ystop = 700
scale = 3

out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

plt.imshow(out_img)
plt.show()

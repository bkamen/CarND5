from TrackingFunctions import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import glob

# Parameters
y_start_stop = [None, None]  # Min and max in y to search in slide_window()
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# load classifier and scaler
svc = joblib.load('CarClassifier.pkl')
X_scaler_color = joblib.load('ScalerColor.pkl')
X_scaler_hog = joblib.load('ScalerHOG.pkl')

xy_window_multiscale = np.array((#[32, 32],
                                 [96, 96],
                                 [128, 128],
                                 [172, 172]))
y_start_stop_multiscale = np.array((#[400, 600],
                                    [400, 600],
                                    [400, 700],
                                    [400, 700]))
x_start_stop_multiscale = np.array((#[400, None],
                                    [400, None],
                                    [400, None],
                                    [400, None]))
xy_overlap_multiscale = np.array((#[0.1, 0.1],
                                  [0.4, 0.4],
                                  [0.6, 0.6],
                                  [0.6, 0.6]))

images = glob.glob('./test_images/*.jpg')

for im in images:
    image = mpimg.imread(im)
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    windows = []
    for i, j, k, l in zip(xy_window_multiscale, y_start_stop_multiscale, x_start_stop_multiscale, xy_overlap_multiscale):
        windows_scale = slide_window(image, k, j, i, l)
        img_ = draw_boxes(draw_image, windows_scale, (255, 0, 0), 1)
        #plt.imshow(img_)
        #plt.show()
        windows.extend(windows_scale)

    hot_windows = search_windows(image, windows, svc, X_scaler_color, X_scaler_hog, color_space=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    mpimg.imsave('./output_images/rectangles_' + im.split('\\')[-1], window_img)

    heatmap = np.zeros_like(image[:, :, 0])
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 0)
    heatmap = np.clip(heatmap, 0, 255)
    mpimg.imsave('./output_images/heatmap_' + im.split('\\')[-1], heatmap, cmap='hot')

    labels = label(heatmap)
    mpimg.imsave('./output_images/label_' + im.split('\\')[-1], labels[0], cmap='gray')
    out_img = draw_labeled_bboxes(draw_image, labels)
    mpimg.imsave('./output_images/final_' + im.split('\\')[-1], out_img)


print('Done.')
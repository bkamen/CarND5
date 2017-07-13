from TrackingFunctions import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from hmap import Hmap

# Parameters
y_start_stop = [None, None]  # Min and max in y to search in slide_window()
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





# load classifier and scaler
svc = joblib.load('CarClassifier.pkl')
X_scaler_color = joblib.load('ScalerColor.pkl')
X_scaler_hog = joblib.load('ScalerHOG.pkl')

#image = mpimg.imread('./test_images/test1.jpg')
#draw_image = np.copy(image)
#image = image.astype(np.float32)/255

xy_window_multiscale = np.array(([96, 96],
                                 [128, 128],
                                 [192, 192]))
y_start_stop_multiscale = np.array(([400, 600],
                                    [400, 700],
                                    [300, None]))
x_start_stop_multiscale = np.array(([400, None],
                                    [400, None],
                                    [300, None]))
xy_overlap_multiscale = np.array(([0.7, 0.7],
                                  [0.7, 0.7],
                                  [0.7, 0.7]))


def detect_cars(image):
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255
    windows = []
    for i, j, k, l in zip(xy_window_multiscale, y_start_stop_multiscale, x_start_stop_multiscale, xy_overlap_multiscale):
        windows_scale = slide_window(image, k, j, i, l)
        windows.extend(windows_scale)

    hot_windows = search_windows(image, windows, svc, X_scaler_color, X_scaler_hog, color_space=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

   # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    heatmap = np.zeros_like(image[:, :, 0])
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = np.clip(apply_threshold(heatmap, 1), 0, 255)

    labels = label(heatmap)
    out_img = draw_labeled_bboxes(draw_image, labels)

    return out_img

vid_output = 'project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4', audio=False)
vid_clip = clip1.fl_image(detect_cars)
vid_clip.write_videofile(vid_output, audio=False)

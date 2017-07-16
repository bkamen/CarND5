##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for the HOG feature extraction is in the file 'TrackingFunctions.py' line 7-24. The single channels of the 'YCrCb' colorspace are passed to the function and evaluated.
Parameters are:
 - 16 pixels per cell
 - 2 cells per block
 - 9 orientation bins


####2. Explain how you settled on your final choice of HOG parameters.

The goal was to have a tradeoff between speed of the program and accuracy score of the classifier.
I started with 32 pixels per cell, 2 cells per block and 9 orientations bins. The only change that significantly changed the accuracy without blowing the feature space up too mich was reducing pixels per cell to 16.
A further reduction to 8 didn't yield much improvement in accuracy but a lot of disadvantage is speed. Same for increasing the orientation bins. Reducing the cells per block resulted in an accuracy drop of the classifier.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature space consists of the HOG features, color histogram data and spatial features.
The parameters are the following:

```python
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
```

For the color histogram and the spatial data the images are also converted to 'YCrCb' colorspace which proved to result in the highest accuracy scores.
The color histogram data is simply a histogram calculation of every color channel

```python
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range, density=True)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range, density=True)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range, density=True)
```

Important is to keep the magnitude of the output in mind since the resulting array is later concatenated with the spatial data.
By setting ``density=True`` the output are values between 0 and 1 and therefore in the same magnitude order as the spatial data using the 'YCrCb' color space.
When a different color space is used the output of the spatial transformation has to be normalized.

The feature vector length for one image of size 64x64 pixels is 1836.

In the script "project_classifier.py" the classifier is trained. The first step is to extract all the above mentioned features (line 34-54).
Here the HoG and the color features are kept separated at first. Reason is that each feature vector is normalized using the sklearn standard scaler ``sklearn.preprocessing.StandardScaler()``.
The reason the feature vectors are not first being concatenated and then normalized is to cover the fact that the values in the feature vectors have different magnitude orders.

Before normalization:

![](./output_images/color_features.png "")
![](./output_images/hog_features.png "")

The HoG values have different magnitudes than the color features.

After normalization:




###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


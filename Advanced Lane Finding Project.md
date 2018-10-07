
# Advanced Lane Finding Project

## Project Objective

Develop an image processing pipeline that can identify lane lines, measure the radius of the curve(if any), identify the center of the lane, determine how far off-center the vehicle is currently driving, and clearly label identfied lines and lane space.





```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import interact, interactive, fixed
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

%matplotlib inline

```

##### Camera Calibration

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for calibrating the camera is located in the cell below. The camera calibration routine works by finding the corner pixels of a pattern with know values and parallelism qualities. The pattern is important to be able to detection distortion in a camera lens. Each of the detected corners is an object point in worldspace. Since we know the true geometry of the objpoints in world space, we can measure any deviation and calculate lens distortion from these values. OpenCV provides the findChessboardCorners and CalibrateCamera functions which perform this function. CalibrateCamera returns the correction coefficcients which can then be applied to new images using the undistort function. Examples of a distorted and an undistorted image are shown below.


```python
#Calibration



# Read in all of the calibration images provided in the camera_cal directory

images = [cv2.imread(file) for file in glob.glob("camera_cal/*.jpg")]

#Prepare an array for storing the object points

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []

# Variables that desribe the amount of corners in the calibration chessboard

nx = 9
ny = 6

# Iterate through the list of calibration images, convert to grayscale and use the opencv2 fucnction
# to find the pixel coordinates for the cheesboard corners.

for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners on the calibration image
    if ret == True:
        # Draw and display the corners
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (9,6), corners, ret)
     
        cv2.imshow('img', image)
        cv2.waitKey(50)
cv2.destroyAllWindows()

#Return the distortion coefficients so they can be used to undistort the images taken with the same camera

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

```

##### Image processing pipeline(Images)

1. An example of an undistorted image is shown below.


```python
# Undistort and image to verify that everything is working correctly

img = mpimg.imread('camera_cal/calibration1.jpg')

undistorted = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

cv2.imwrite('UndistortedImage-test2.jpg',undistorted)
```




    True




![png](output_5_1.png)



```python
#### My pseudo Code for the project ######

#get lens calibration values
#slice video into images
#apply lens calibration to video images
#warp video images so they display an overhead view
#Figure out what colorspace I want to used to detect lane lines(simplecv tool could be useful here)
#Take colorspace shifted images and use rectangle histogram method to identfy location of lane lines
#Shift rectangle upwards and repeat until completed for the whole image
#take lane line locations and generate a polyfit curve that best fits through their centers
#Draw the polyfit line in yellow
#Draw a rectange the space between the lines in grew
#calculate the center of the lane
#calculate the distance of the vehicle from center
#calculate the radius of the curve
#dewarp the resultant image with identified lane lines
#display the calculated radius, centerm and distance from center
#Repeat for all images
#Recreate video from processed images
```

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

My image processing pipeline made extensive use of gradient and color threhold binary images in order to create a robust outline of the lane lines. Colorspaces used included the RGB(R Channel), HLS(S Channel), and LUV(L Channel). The RGB channel was used to detect yellow lines. The S Channel was used to detect both white and yellow lines in bright lighting conditions. The L channel was used to detect lane lines in dark lighting conditions. Each of the color channel images was converted to a binary image to aid in lane line detection further on in the pipeline. In addition to the Color transforms that were applied, Sobel(X and Y) and Magnitude gradients were used to further detect lane line pixels. In the final pipeline I used combinational logic(AND and OR) to provide a stronger lane line pixel detection process. Examples of each of these transforms and the subsequent combinations are show in the next series of code cells. 



```python
################# S Channel Color transform ###########################

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

################# L Channel Color transform ###########################

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l,u,v = cv2.split(luv)
    binary_output = np.zeros_like(l)
    binary_output[(l > thresh[0]) & (l <= thresh[1])] = 1
    return binary_output

################# R Channel Color transform ###########################

def rgb_select(img, thresh=(0, 255)):
    r,g,b = cv2.split(img)
    binary_output = np.zeros_like(r)
    binary_output[(r > thresh[0]) & (r <= thresh[1])] = 1
    return binary_output

################# Absolute Gradient Binary threhold transform ###########################

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

################# Magnitude Gradient Binary threhold transform ###########################

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
```


```python
# Read in the images provided for pipeline testing as part of the project directory

img1 = mpimg.imread('test_images/test1.jpg')
img2 = mpimg.imread('test_images/test2.jpg')
img3 = mpimg.imread('test_images/test3.jpg')
img4 = mpimg.imread('test_images/test4.jpg')
img5 = mpimg.imread('test_images/test5.jpg')
img6 = mpimg.imread('test_images/test6.jpg')
img7 = mpimg.imread('test_images/straight_lines1.jpg')
img8 = mpimg.imread('test_images/straight_lines2.jpg')

undistorted1 = cv2.undistort(img1, mtx, dist, None, mtx)
undistorted2 = cv2.undistort(img2, mtx, dist, None, mtx)
undistorted3 = cv2.undistort(img3, mtx, dist, None, mtx)
undistorted4 = cv2.undistort(img4, mtx, dist, None, mtx)
undistorted5 = cv2.undistort(img5, mtx, dist, None, mtx)
undistorted6 = cv2.undistort(img6, mtx, dist, None, mtx)
```


```python
################# Apply the R Channel transform to the test images ###########################

min = 220
max = 255

r_channel1 = rgb_select(undistorted1, thresh=(min, max))
r_channel2 = rgb_select(undistorted2, thresh=(min, max))
r_channel3 = rgb_select(undistorted3, thresh=(min, max))
r_channel4 = rgb_select(undistorted4, thresh=(min, max))
r_channel5 = rgb_select(undistorted5, thresh=(min, max))
r_channel6 = rgb_select(undistorted6, thresh=(min, max))
    
f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(r_channel1, cmap='gray')
x1.set_title('RGB R Channel Test Image 1', fontsize=50)
x2.imshow(r_channel2, cmap='gray')
x2.set_title('RGB R Channel Test Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d8cd07668>




![png](output_10_1.png)



```python




upper = 255
lower = 180

hls_binary1 = hls_select(undistorted1, thresh=(lower,upper))
hls_binary2 = hls_select(undistorted2, thresh=(lower,upper))
hls_binary3 = hls_select(undistorted3, thresh=(lower,upper))
hls_binary4 = hls_select(undistorted4, thresh=(lower,upper))
hls_binary5 = hls_select(undistorted5, thresh=(lower,upper))
hls_binary6 = hls_select(undistorted6, thresh=(lower,upper))



f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(hls_binary1, cmap='gray')
x1.set_title('HLS S Channel Test Image 1', fontsize=50)
x2.imshow(hls_binary2, cmap='gray')
x2.set_title('HLS S Channel Test Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d8f66b320>




![png](output_11_1.png)



```python
lower_th = 200
upper_th = 230


luv_binary1 = luv_select(undistorted1, thresh=(lower_th, upper_th))
luv_binary2 = luv_select(undistorted2, thresh=(lower_th, upper_th))
luv_binary3 = luv_select(undistorted3, thresh=(lower_th, upper_th))
luv_binary4 = luv_select(undistorted4, thresh=(lower_th, upper_th))
luv_binary5 = luv_select(undistorted5, thresh=(lower_th, upper_th))
luv_binary6 = luv_select(undistorted6, thresh=(lower_th, upper_th))

f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(luv_binary1, cmap='gray')
x1.set_title('LUV L Channel Test Image 1', fontsize=50)
x2.imshow(luv_binary2, cmap='gray')
x2.set_title('LUV L Channel Test Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d8ff26f28>




![png](output_12_1.png)



```python

min = 20
max = 200
grad_binary1x = abs_sobel_thresh(undistorted1, orient='x', thresh_min=min, thresh_max=max)
grad_binary2x = abs_sobel_thresh(undistorted2, orient='x', thresh_min=min, thresh_max=max)
grad_binary3x = abs_sobel_thresh(undistorted3, orient='x', thresh_min=min, thresh_max=max)
grad_binary4x = abs_sobel_thresh(undistorted4, orient='x', thresh_min=min, thresh_max=max)
grad_binary5x = abs_sobel_thresh(undistorted5, orient='x', thresh_min=min, thresh_max=max)
grad_binary6x = abs_sobel_thresh(undistorted, orient='x', thresh_min=min, thresh_max=max)


f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(grad_binary1x, cmap='gray')
x1.set_title('Sobel X Image 1', fontsize=50)
x2.imshow(grad_binary2x, cmap='gray')
x2.set_title('Sobel X Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d907eaf98>




![png](output_13_1.png)



```python
min = 20
max = 200

grad_binary1y = abs_sobel_thresh(undistorted1, orient='y', thresh_min=min, thresh_max=max)
grad_binary2y = abs_sobel_thresh(undistorted2, orient='y', thresh_min=min, thresh_max=max)
grad_binary3y = abs_sobel_thresh(undistorted3, orient='y', thresh_min=min, thresh_max=max)
grad_binary4y = abs_sobel_thresh(undistorted4, orient='y', thresh_min=min, thresh_max=max)
grad_binary5y = abs_sobel_thresh(undistorted5, orient='y', thresh_min=min, thresh_max=max)
grad_binary6y = abs_sobel_thresh(undistorted6, orient='y', thresh_min=min, thresh_max=max)


f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(grad_binary1y, cmap='gray')
x1.set_title('Sobel Y Image 1', fontsize=50)
x2.imshow(grad_binary2y, cmap='gray')
x2.set_title('Sobel Y Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d910b3dd8>




![png](output_14_1.png)



```python
min = 50
max = 160
mag_binary1 = mag_thresh(undistorted1, sobel_kernel=3, mag_thresh=(min, max))
mag_binary2 = mag_thresh(undistorted2, sobel_kernel=3, mag_thresh=(min, max))
mag_binary3 = mag_thresh(undistorted3, sobel_kernel=3, mag_thresh=(min, max))
mag_binary4 = mag_thresh(undistorted4, sobel_kernel=3, mag_thresh=(min, max))
mag_binary5 = mag_thresh(undistorted5, sobel_kernel=3, mag_thresh=(min, max))
mag_binary6 = mag_thresh(undistorted6, sobel_kernel=3, mag_thresh=(min, max))


f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(mag_binary1, cmap='gray')
x1.set_title('Mag Thresh Image 1', fontsize=50)
x2.imshow(mag_binary2, cmap='gray')
x2.set_title('Mag Thresh Image 2', fontsize=50)

```




    <matplotlib.text.Text at 0x24d915dcc18>




![png](output_15_1.png)



```python
and_image1 = cv2.bitwise_and(luv_binary1, hls_binary1)
and_image2 = cv2.bitwise_and(luv_binary2, hls_binary2)
and_image3 = cv2.bitwise_and(luv_binary3, hls_binary3)
and_image4 = cv2.bitwise_and(luv_binary4, hls_binary4)
and_image5 = cv2.bitwise_and(luv_binary5, hls_binary5)
and_image6 = cv2.bitwise_and(luv_binary6, hls_binary6)

sobel_and1 = cv2.bitwise_and(grad_binary1x, grad_binary1y)
sobel_and2 = cv2.bitwise_and(grad_binary2x, grad_binary2y)
sobel_and3 = cv2.bitwise_and(grad_binary3x, grad_binary3y)
sobel_and4 = cv2.bitwise_and(grad_binary4x, grad_binary4y)
sobel_and5 = cv2.bitwise_and(grad_binary5x, grad_binary5y)
sobel_and6 = cv2.bitwise_and(grad_binary6x, grad_binary6y)

and1 = cv2.bitwise_and(mag_binary1, r_channel1)
and2 = cv2.bitwise_and(mag_binary2, r_channel2)
and3 = cv2.bitwise_and(mag_binary3, r_channel3)
and4 = cv2.bitwise_and(mag_binary4, r_channel4)
and5 = cv2.bitwise_and(mag_binary5, r_channel5)
and6 = cv2.bitwise_and(mag_binary6, r_channel6)

or_image1 = cv2.bitwise_or(and_image1, sobel_and1, and1)
or_image2 = cv2.bitwise_or(and_image2, sobel_and2, and2)
or_image3 = cv2.bitwise_or(and_image3, sobel_and3, and3)
or_image4 = cv2.bitwise_or(and_image4, sobel_and4, and4)
or_image5 = cv2.bitwise_or(and_image5, sobel_and5, and5)
or_image6 = cv2.bitwise_or(and_image6, sobel_and6, and6)

f1, (x1, x2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()
x1.imshow(or_image1, cmap='gray')
x1.set_title('Final Transform Image 1', fontsize=50)
x2.imshow(or_image2, cmap='gray')
x2.set_title('Final Transform  Image 2', fontsize=50)


```




    <matplotlib.text.Text at 0x24d931f67b8>




![png](output_16_1.png)


##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code below applies a perspective transform to the lane region. The SRC points are manually selected from an image where the lanes are straight. The destination points are chosen to form a rectangle as we want to take the normal view and create a new "top down" view. The getPerspectiveTransform function is used to both warp and unwarp the images. The warpPerspective function applies the previously generated M or M_inv warp coefficients to an image.


```python
# Select Source Points
src_bottom_left = [570,468] 
src_bottom_right = [714,468]
src_top_left = [207,720]
src_top_right = [1106,720]

source = np.float32([src_bottom_left,src_bottom_right,src_top_right,src_top_left])

image_shape = (720,1280)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720] 
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]

dst = np.float32([top_left,top_right,bottom_right, bottom_left])

M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (image_shape[1], image_shape[0])

warped1 = cv2.warpPerspective(or_image1, M, img_size , flags=cv2.INTER_LINEAR)
warped2 = cv2.warpPerspective(or_image2, M, img_size , flags=cv2.INTER_LINEAR)
warped3 = cv2.warpPerspective(or_image3, M, img_size , flags=cv2.INTER_LINEAR)
warped4 = cv2.warpPerspective(or_image4, M, img_size , flags=cv2.INTER_LINEAR)
warped5 = cv2.warpPerspective(or_image5, M, img_size , flags=cv2.INTER_LINEAR)
warped6 = cv2.warpPerspective(or_image6, M, img_size , flags=cv2.INTER_LINEAR)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(or_image1, cmap='gray')
ax1.set_title('Image 1', fontsize=50)
ax2.imshow(warped1, cmap='gray')
ax2.set_title('Warped Image 1', fontsize=50)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(or_image2, cmap='gray')
ax1.set_title('Image 2', fontsize=50)
ax2.imshow(warped2, cmap='gray')
ax2.set_title('Warped  Image 2', fontsize=50)



```




    <matplotlib.text.Text at 0x24d96f342b0>




![png](output_18_1.png)



![png](output_18_2.png)


##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line pixels are using the method outlined in the lecture. The algorithm can be broken down to the following basic steps.

1. Take a histogram of the bottom half of a top down view of the lane line binary image.
2. Detect the peak in the left half and right half of the image. 
3. Use the x location of the left and right peaks to begin a more detailed scan for the lane lines starting from the bottom of the image.
4. Decide how many scan steps you would like to perform in the y direction. This is hardcoded and equal to (Image Height/Scan Steps) = Scan Height. Scan Width is Fixed at 100.
5. Within this rectangle count how many white pixels there are. If there are more than fifty then a lane line is considered detected, and the centroid is calculated. 
6. The rectangle is drawn around the white pixel blob centroid.
7. This is repeated all the way to the top of the picture.
8. Use a polyfit function to defined a polyfit line through the center of the left and right rectangle centroids.

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature of the line can be found using the formula

![alt text](RoC.jpg "Radius of Curvature Formula")

This is coded in python as

###### ((1 + (2*left_fitcr[0]*y_eval*ym_per_pix + left_fitcr[1])**2)**1.5) / np.absolute(2*left_fitcr[0])

Ym_per_pix is a constant value that describes meters in the Y direction per pixel. This value is a suggested value from the Udacity lecture.

The radius of the curve is calculated and displayed in an image below.

The center of the image is assumed to be the center of the car. The center of the lane is calculated by finding the distance between the left and right detected lanes. Once we have identified the center lane pixel, we can calculate the distance the lane center is from the image center and multiply the difference in pixel by Yx_per_pix. Yx_per_pix describes how many meters in the x direction a pixel represents. The calculated center offset is located in the resultant image a couple of cells down.


6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

An example with an identified lane(green) as well as left and right lane radius and center offset displayed is located below.


```python
histogram = np.sum(warped2[warped2.shape[0]//2:,:], axis=0)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

ax1.imshow(warped2, cmap='gray')
ax2.plot(histogram)
```




    [<matplotlib.lines.Line2D at 0x24d97bca668>]




![png](output_20_1.png)



```python


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[100:midpoint])
    rightx_base = np.argmax(histogram[midpoint:1100]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

left_fitx_hist=[]
right_fitx_hist=[]

def fit_polynomial(binary_warped, Minv, undist):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    global left_fitx_hist
    left_fitx_hist.append(left_fitx)
    
    var = 120
   
    if len(left_fitx_hist)>var:
        left_fitx_hist=left_fitx_hist[-(var-1):]
    left_fitx_hist.append(left_fitx)
        
        
    #Build right fit history in order to identify outliers
    
    global right_fitx_hist
    right_fitx_hist.append(right_fitx)
    
    if len(right_fitx_hist)>var:
        right_fitx_hist=right_fitx_hist[-(var-1):]
    right_fitx_hist.append(right_fitx)
    

    left_fitx_mean = np.mean(np.array(left_fitx_hist), axis=0 )[:len(ploty)]
    right_fitx_mean = np.mean(np.array(right_fitx_hist), axis=0 )[:len(ploty)] 


    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    #Caclculate the radius of the curves
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700 
    left_fitcr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fitcr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    left_curve_rad = ((1 + (2*left_fitcr[0]*y_eval*ym_per_pix + left_fitcr[1])**2)**1.5) / np.absolute(2*left_fitcr[0])
    right_curve_rad = ((1 + (2*right_fitcr[0]*y_eval*ym_per_pix + right_fitcr[1])**2)**1.5) / np.absolute(2*right_fitcr[0])
    print("The radius of curvature is:", left_curve_rad, "m", right_curve_rad, "m")

    #calculate the fill poly area between the lane lines
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx_mean, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_mean, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.8, 0)

    
    # Add text to an image
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    topline_text = 'Curvature: L %.3fm  R %.3fm' % (left_curve_rad, right_curve_rad)

    # how to calculate camera offset
    camera_position = image.shape[1]/2
    lane_center = (right_fitx[719] + left_fitx[719])/2
    center_offset_pixels = camera_position - lane_center
    center_offset_dist = center_offset_pixels * xm_per_pix

    bottomline_text = 'Offset: %.3fm' % (center_offset_dist)
    
    cv2.putText(result,topline_text,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,bottomline_text,(10,100), font, 1, (255,255,255),2,cv2.LINE_AA)
    
    return result



```


```python
result= fit_polynomial(warped5, M_inv, undistorted5)

plt.imshow(result)
```

    The radius of curvature is: 1056.6127170955456 m 4213.772934392879 m
    




    <matplotlib.image.AxesImage at 0x24d97c1bc18>




![png](output_22_2.png)



```python
# Select Source Points
src_bottom_left = [570,468] 
src_bottom_right = [714,468]
src_top_left = [207,720]
src_top_right = [1106,720]

source = np.float32([src_bottom_left,src_bottom_right,src_top_right,src_top_left])

image_shape = (720,1280)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720] 
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]
    
dst = np.float32([top_left,top_right,bottom_right, bottom_left])
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (image_shape[1], image_shape[0])

def pipeline(img):
    
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    hls_binary = hls_select(undistorted, thresh=(180, 255))
    r_channel = rgb_select(img, thresh=(220, 255))
    grad_binary_x =  abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100) 
    grad_binary_y =  abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 255))
    luv_binary = luv_select(undistorted1, thresh=(200, 230))
    
    and_image1 = cv2.bitwise_and(hls_binary, luv_binary)
    sobel_and1 = cv2.bitwise_and(grad_binary_x, grad_binary_y)
    and1 = cv2.bitwise_and(r_channel, mag_binary)
    
    
    or_image =cv2.bitwise_or(and_image1, sobel_and1, and1)
    
    
    #or_image = cv2.bitwise_or(or_image0, and_image, r_channel)
    
    warped = cv2.warpPerspective(or_image, M, img_size , flags=cv2.INTER_LINEAR)
    res = fit_polynomial(warped, M_inv, undistorted)
    
    return res
```


```python
testout1 = pipeline(img1)
testout2 = pipeline(img2)
testout3 = pipeline(img3)
testout4 = pipeline(img4)
testout5 = pipeline(img5)
testout6 = pipeline(img6)
testout7 = pipeline(img7)
testout8 = pipeline(img8)

plt.imshow(testout1)
mpimg.imsave('testout1.jpg',testout1)
mpimg.imsave('testout2.jpg',testout2)
mpimg.imsave('testout3.jpg',testout3)
mpimg.imsave('testout4.jpg',testout4)
mpimg.imsave('testout5.jpg',testout5)
mpimg.imsave('testout6.jpg',testout6)
mpimg.imsave('testout7.jpg',testout7)
mpimg.imsave('testout8.jpg',testout8)
```

    The radius of curvature is: 1436.0305862021164 m 1545.7808318261561 m
    The radius of curvature is: 1386.6728679039797 m 427.6736833574265 m
    The radius of curvature is: 2821.0113334771945 m 1293.6991948707332 m
    The radius of curvature is: 1958.3521268478735 m 175.48305050896425 m
    The radius of curvature is: 325.0216640249139 m 11366.433826871464 m
    The radius of curvature is: 1776.0550272948865 m 4386.997575624305 m
    The radius of curvature is: 4380.2620605129005 m 34332.39756633411 m
    The radius of curvature is: 5615.433450902887 m 17798.62301971316 m
    


![png](output_24_1.png)


##### Pipeline (video)

1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a link to my final video hosted on youtube.[My final video](https://youtu.be/iA6K0D77vGk "My final video")

This video can be recreated by running all of the cells within the Advanced Lane Finding Project.ipynb from within the standard udacity project repo.



```python
yellow_output = 'testvideo.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('project_video.mp4')
yellow_clip = clip2.fl_image(pipeline)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    The radius of curvature is: 956.2246195828358 m 6044.360454100552 m
    [MoviePy] >>>> Building video testvideo.mp4
    [MoviePy] Writing video testvideo.mp4
    

      0%|                                                                                         | 0/1261 [00:00<?, ?it/s]

    The radius of curvature is: 956.2246195828358 m 6044.360454100552 m
    

      0%|                                                                                 | 1/1261 [00:00<03:01,  6.93it/s]

    The radius of curvature is: 1056.1217966392633 m 8558.771233345662 m
    

      0%|▏                                                                                | 2/1261 [00:00<03:06,  6.75it/s]

    The radius of curvature is: 968.6144467885244 m 2502.3587169611446 m
    

      0%|▏                                                                                | 3/1261 [00:00<03:06,  6.74it/s]

    The radius of curvature is: 831.368469820884 m 1074.4158052213436 m
    

      0%|▎                                                                                | 4/1261 [00:00<03:05,  6.76it/s]

    The radius of curvature is: 996.7869230286362 m 2087.2245693184123 m
    

      0%|▎                                                                                | 5/1261 [00:00<03:06,  6.74it/s]

    The radius of curvature is: 1070.4682469764 m 2377.258602028977 m
    

      0%|▍                                                                                | 6/1261 [00:00<03:06,  6.73it/s]

    The radius of curvature is: 1165.1238829442698 m 1577.8231692413733 m
    

      1%|▍                                                                                | 7/1261 [00:01<03:06,  6.73it/s]

    The radius of curvature is: 1165.6938499042988 m 584.6751166180402 m
    

      1%|▌                                                                                | 8/1261 [00:01<03:06,  6.73it/s]

    The radius of curvature is: 1269.5269475999578 m 932.7590667312779 m
    

      1%|▌                                                                                | 9/1261 [00:01<03:05,  6.74it/s]

    The radius of curvature is: 1008.6427723217245 m 1465.5171196720676 m
    

      1%|▋                                                                               | 10/1261 [00:01<03:09,  6.62it/s]

    The radius of curvature is: 973.9568282744123 m 1948.1880180790724 m
    

      1%|▋                                                                               | 11/1261 [00:01<03:09,  6.60it/s]

    The radius of curvature is: 773.1908192288056 m 1684.4665009720557 m
    

      1%|▊                                                                               | 12/1261 [00:01<03:09,  6.60it/s]

    The radius of curvature is: 933.5841097747333 m 11561.994769013556 m
    

      1%|▊                                                                               | 13/1261 [00:01<03:08,  6.62it/s]

    The radius of curvature is: 839.7732484736555 m 1081.2202190651815 m
    

      1%|▉                                                                               | 14/1261 [00:02<03:07,  6.64it/s]

    The radius of curvature is: 855.8443895841908 m 2914.2544770898458 m
    

      1%|▉                                                                               | 15/1261 [00:02<03:07,  6.66it/s]

    The radius of curvature is: 658.5139185413157 m 1964.9781678373265 m
    

      1%|█                                                                               | 16/1261 [00:02<03:06,  6.68it/s]

    The radius of curvature is: 1180.552240779936 m 6044.513611168531 m
    

      1%|█                                                                               | 17/1261 [00:02<03:05,  6.70it/s]

    The radius of curvature is: 161.27314272122518 m 4786.632531016335 m
    

      1%|█▏                                                                              | 18/1261 [00:02<03:05,  6.71it/s]

    The radius of curvature is: 1619.104998626433 m 59411.252089884416 m
    

      2%|█▏                                                                              | 19/1261 [00:02<03:04,  6.71it/s]

    The radius of curvature is: 1942.592910280058 m 1442.5700793794883 m
    

      2%|█▎                                                                              | 20/1261 [00:02<03:04,  6.72it/s]

    The radius of curvature is: 58.904271597267396 m 916.0712446749546 m
    

      2%|█▎                                                                              | 21/1261 [00:03<03:04,  6.74it/s]

    The radius of curvature is: 177.8401357386645 m 569.8914678718227 m
    

      2%|█▍                                                                              | 22/1261 [00:03<03:04,  6.70it/s]

    The radius of curvature is: 900.6766517682511 m 977.4450112132341 m
    

      2%|█▍                                                                              | 23/1261 [00:03<03:04,  6.69it/s]

    The radius of curvature is: 236.0279358442115 m 7836.750124980482 m
    

      2%|█▌                                                                              | 24/1261 [00:03<03:05,  6.67it/s]

    The radius of curvature is: 302.4186599891084 m 1509.572868278811 m
    

      2%|█▌                                                                              | 25/1261 [00:03<03:05,  6.65it/s]

    The radius of curvature is: 493.2743504584666 m 1847.793723315274 m
    

      2%|█▋                                                                              | 26/1261 [00:03<03:05,  6.65it/s]

    The radius of curvature is: 898.4551784775095 m 5987.758578815796 m
    

      2%|█▋                                                                              | 27/1261 [00:04<03:05,  6.65it/s]

    The radius of curvature is: 932.5094746888054 m 1906.797486909845 m
    

      2%|█▊                                                                              | 28/1261 [00:04<03:05,  6.65it/s]

    The radius of curvature is: 1243.498352746765 m 6468.2150083575125 m
    

      2%|█▊                                                                              | 29/1261 [00:04<03:05,  6.65it/s]

    The radius of curvature is: 1187.4112499113892 m 1162.2670342262993 m
    

      2%|█▉                                                                              | 30/1261 [00:04<03:04,  6.66it/s]

    The radius of curvature is: 1180.520208639987 m 2361.565398193821 m
    

      2%|█▉                                                                              | 31/1261 [00:04<03:04,  6.66it/s]

    The radius of curvature is: 1063.405537733089 m 2023.0135066744626 m
    

      3%|██                                                                              | 32/1261 [00:04<03:04,  6.67it/s]

    The radius of curvature is: 1226.611782549901 m 1168.8327302657408 m
    

      3%|██                                                                              | 33/1261 [00:04<03:04,  6.65it/s]

    The radius of curvature is: 1174.640329961945 m 520.3025605630938 m
    

      3%|██▏                                                                             | 34/1261 [00:05<03:04,  6.65it/s]

    The radius of curvature is: 1166.6572250406275 m 753.22655070487 m
    

      3%|██▏                                                                             | 35/1261 [00:05<03:04,  6.63it/s]

    The radius of curvature is: 806.3223433735515 m 1233.5981739783642 m
    

      3%|██▎                                                                             | 36/1261 [00:05<03:04,  6.63it/s]

    The radius of curvature is: 792.3965945078771 m 1639.5279392679831 m
    

      3%|██▎                                                                             | 37/1261 [00:05<03:05,  6.61it/s]

    The radius of curvature is: 710.3452236738426 m 1652.9889923532846 m
    

      3%|██▍                                                                             | 38/1261 [00:05<03:04,  6.61it/s]

    The radius of curvature is: 682.7290887416038 m 2481.9929706270673 m
    

      3%|██▍                                                                             | 39/1261 [00:05<03:04,  6.61it/s]

    The radius of curvature is: 734.6587467879825 m 1707.8286445810697 m
    

      3%|██▌                                                                             | 40/1261 [00:06<03:04,  6.60it/s]

    The radius of curvature is: 725.5301576386889 m 994.1788207347803 m
    

      3%|██▌                                                                             | 41/1261 [00:06<03:04,  6.60it/s]

    The radius of curvature is: 657.1693367957304 m 1516.6756177041493 m
    

      3%|██▋                                                                             | 42/1261 [00:06<03:04,  6.61it/s]

    The radius of curvature is: 779.3782217946583 m 1819.9090856628954 m
    

      3%|██▋                                                                             | 43/1261 [00:06<03:04,  6.61it/s]

    The radius of curvature is: 854.4185236205591 m 2068.195815009451 m
    

      3%|██▊                                                                             | 44/1261 [00:06<03:04,  6.61it/s]

    The radius of curvature is: 883.3137462058261 m 2515.3966728469577 m
    

      4%|██▊                                                                             | 45/1261 [00:06<03:03,  6.61it/s]

    The radius of curvature is: 816.0754478136707 m 776.8782278267269 m
    

      4%|██▉                                                                             | 46/1261 [00:06<03:03,  6.62it/s]

    The radius of curvature is: 754.579827781357 m 733.0219481754532 m
    

      4%|██▉                                                                             | 47/1261 [00:07<03:03,  6.62it/s]

    The radius of curvature is: 913.4820197061159 m 638.2371406781472 m
    

      4%|███                                                                             | 48/1261 [00:07<03:02,  6.63it/s]

    The radius of curvature is: 765.918320720342 m 793.4245727787969 m
    

      4%|███                                                                             | 49/1261 [00:07<03:02,  6.63it/s]

    The radius of curvature is: 863.0312091598106 m 1196.6828175736748 m
    

      4%|███▏                                                                            | 50/1261 [00:07<03:02,  6.63it/s]

    The radius of curvature is: 890.8167553579403 m 3356.454023340939 m
    

      4%|███▏                                                                            | 51/1261 [00:07<03:02,  6.64it/s]

    The radius of curvature is: 796.4258454315003 m 3294.7828434933494 m
    

      4%|███▎                                                                            | 52/1261 [00:07<03:02,  6.63it/s]

    The radius of curvature is: 825.6856644018982 m 36832.597436831784 m
    

      4%|███▎                                                                            | 53/1261 [00:07<03:01,  6.64it/s]

    The radius of curvature is: 1073.8953293705442 m 854.8146954260166 m
    

      4%|███▍                                                                            | 54/1261 [00:08<03:01,  6.64it/s]

    The radius of curvature is: 1474.9046904399975 m 1247.9165805169237 m
    

      4%|███▍                                                                            | 55/1261 [00:08<03:01,  6.64it/s]

    The radius of curvature is: 1892.4984445634489 m 1891.9469055509437 m
    

      4%|███▌                                                                            | 56/1261 [00:08<03:01,  6.64it/s]

    The radius of curvature is: 3098.4989031685113 m 1352.5993524303917 m
    

      5%|███▌                                                                            | 57/1261 [00:08<03:01,  6.63it/s]

    The radius of curvature is: 2270.8073850528826 m 737.0389395935101 m
    

      5%|███▋                                                                            | 58/1261 [00:08<03:01,  6.62it/s]

    The radius of curvature is: 1906.1120240867413 m 509.3679873697717 m
    

      5%|███▋                                                                            | 59/1261 [00:08<03:01,  6.62it/s]

    The radius of curvature is: 1663.6004129519938 m 872.2992271408544 m
    

      5%|███▊                                                                            | 60/1261 [00:09<03:01,  6.62it/s]

    The radius of curvature is: 1297.1711616473187 m 779.0105447523605 m
    

      5%|███▊                                                                            | 61/1261 [00:09<03:01,  6.62it/s]

    The radius of curvature is: 1202.0882999721086 m 1254.5663807997892 m
    

      5%|███▉                                                                            | 62/1261 [00:09<03:00,  6.63it/s]

    The radius of curvature is: 887.8710319091689 m 4242.7307655825525 m
    

      5%|███▉                                                                            | 63/1261 [00:09<03:00,  6.63it/s]

    The radius of curvature is: 848.9663848767199 m 2304.0757629095 m
    

      5%|████                                                                            | 64/1261 [00:09<03:00,  6.63it/s]

    The radius of curvature is: 951.7300669574948 m 1677.6657119896306 m
    

      5%|████                                                                            | 65/1261 [00:09<03:00,  6.63it/s]

    The radius of curvature is: 983.1173597777175 m 2702.4172051967867 m
    

      5%|████▏                                                                           | 66/1261 [00:09<03:00,  6.63it/s]

    The radius of curvature is: 946.1953026121391 m 20968.077863823924 m
    

      5%|████▎                                                                           | 67/1261 [00:10<03:00,  6.63it/s]

    The radius of curvature is: 929.7768652741493 m 16536.939623870727 m
    

      5%|████▎                                                                           | 68/1261 [00:10<02:59,  6.64it/s]

    The radius of curvature is: 880.8051682852398 m 2316.998777119632 m
    

      5%|████▍                                                                           | 69/1261 [00:10<02:59,  6.64it/s]

    The radius of curvature is: 987.93894770518 m 2796.819944644496 m
    

      6%|████▍                                                                           | 70/1261 [00:10<02:59,  6.64it/s]

    The radius of curvature is: 1024.3148805469107 m 636.8356999024862 m
    

      6%|████▌                                                                           | 71/1261 [00:10<02:59,  6.64it/s]

    The radius of curvature is: 1070.2669902675946 m 1231.2542309850141 m
    

      6%|████▌                                                                           | 72/1261 [00:10<02:58,  6.64it/s]

    The radius of curvature is: 1161.225823056356 m 1072.5633104547994 m
    

      6%|████▋                                                                           | 73/1261 [00:10<02:58,  6.65it/s]

    The radius of curvature is: 1463.0494554170239 m 936.7099593511751 m
    

      6%|████▋                                                                           | 74/1261 [00:11<02:58,  6.65it/s]

    The radius of curvature is: 1658.1701384848677 m 2635.298921858705 m
    

      6%|████▊                                                                           | 75/1261 [00:11<02:58,  6.65it/s]

    The radius of curvature is: 1877.9892215510429 m 1240.5237147659716 m
    

      6%|████▊                                                                           | 76/1261 [00:11<02:58,  6.65it/s]

    The radius of curvature is: 2024.6067901219528 m 1728.1991230219323 m
    

      6%|████▉                                                                           | 77/1261 [00:11<02:57,  6.65it/s]

    The radius of curvature is: 1942.953338790206 m 1421.942175920277 m
    

      6%|████▉                                                                           | 78/1261 [00:11<02:57,  6.65it/s]

    The radius of curvature is: 2319.0028357474907 m 1237.5001276708902 m
    

      6%|█████                                                                           | 79/1261 [00:11<02:57,  6.65it/s]

    The radius of curvature is: 2181.1121645561448 m 1232.986673443871 m
    

      6%|█████                                                                           | 80/1261 [00:12<02:57,  6.65it/s]

    The radius of curvature is: 5087.2846951882375 m 3928.2483807953804 m
    

      6%|█████▏                                                                          | 81/1261 [00:12<02:57,  6.65it/s]

    The radius of curvature is: 6237.262348115288 m 2351.0118741561464 m
    

      7%|█████▏                                                                          | 82/1261 [00:12<02:57,  6.65it/s]

    The radius of curvature is: 6219.357161729191 m 2197.5326601639185 m
    

      7%|█████▎                                                                          | 83/1261 [00:12<02:56,  6.66it/s]

    The radius of curvature is: 7985.861885337695 m 2345.364321827126 m
    

      7%|█████▎                                                                          | 84/1261 [00:12<02:56,  6.66it/s]

    The radius of curvature is: 10042.31300485814 m 868.019693301784 m
    

      7%|█████▍                                                                          | 85/1261 [00:12<02:56,  6.66it/s]

    The radius of curvature is: 3797.4983112276163 m 730.2440145781508 m
    

      7%|█████▍                                                                          | 86/1261 [00:12<02:56,  6.66it/s]

    The radius of curvature is: 2533.8655776798764 m 1030.0231238259134 m
    

      7%|█████▌                                                                          | 87/1261 [00:13<02:56,  6.66it/s]

    The radius of curvature is: 1916.3165726970465 m 10336.299186841377 m
    

      7%|█████▌                                                                          | 88/1261 [00:13<02:56,  6.66it/s]

    The radius of curvature is: 1481.5750545631104 m 11849.890129705876 m
    

      7%|█████▋                                                                          | 89/1261 [00:13<02:55,  6.67it/s]

    The radius of curvature is: 1431.279118306501 m 1224.9737319654591 m
    

      7%|█████▋                                                                          | 90/1261 [00:13<02:55,  6.67it/s]

    The radius of curvature is: 1268.3711966962799 m 1190.5053174845102 m
    

      7%|█████▊                                                                          | 91/1261 [00:13<02:55,  6.67it/s]

    The radius of curvature is: 1193.953808569384 m 1648.1645642426065 m
    

      7%|█████▊                                                                          | 92/1261 [00:13<02:55,  6.67it/s]

    The radius of curvature is: 1298.364117222025 m 3446.4101483077225 m
    

      7%|█████▉                                                                          | 93/1261 [00:13<02:55,  6.67it/s]

    The radius of curvature is: 1262.0402815795708 m 2362.737459022887 m
    

      7%|█████▉                                                                          | 94/1261 [00:14<02:54,  6.67it/s]

    The radius of curvature is: 1135.4580635310035 m 2078.094916038272 m
    

      8%|██████                                                                          | 95/1261 [00:14<02:54,  6.67it/s]

    The radius of curvature is: 1127.2292323688707 m 728.9648238531659 m
    

      8%|██████                                                                          | 96/1261 [00:14<02:54,  6.67it/s]

    The radius of curvature is: 1048.1637975663384 m 903.3703851078067 m
    

      8%|██████▏                                                                         | 97/1261 [00:14<02:54,  6.67it/s]

    The radius of curvature is: 1267.8051659752655 m 920.6049657049872 m
    

      8%|██████▏                                                                         | 98/1261 [00:14<02:54,  6.67it/s]

    The radius of curvature is: 1168.3115842365053 m 928.2348066876381 m
    

      8%|██████▎                                                                         | 99/1261 [00:14<02:54,  6.68it/s]

    The radius of curvature is: 1235.7407263103173 m 947.9059855703273 m
    

      8%|██████▎                                                                        | 100/1261 [00:14<02:53,  6.68it/s]

    The radius of curvature is: 1163.1293119846673 m 1382.2197212520832 m
    

      8%|██████▎                                                                        | 101/1261 [00:15<02:53,  6.68it/s]

    The radius of curvature is: 1171.5776592122966 m 962.0554882289662 m
    

      8%|██████▍                                                                        | 102/1261 [00:15<02:53,  6.68it/s]

    The radius of curvature is: 1204.0338902887147 m 21808.66109169596 m
    

      8%|██████▍                                                                        | 103/1261 [00:15<02:53,  6.68it/s]

    The radius of curvature is: 1173.9168014565607 m 1143.9250699042514 m
    

      8%|██████▌                                                                        | 104/1261 [00:15<02:53,  6.68it/s]

    The radius of curvature is: 1183.9197304952122 m 1152.830513276434 m
    

      8%|██████▌                                                                        | 105/1261 [00:15<02:53,  6.68it/s]

    The radius of curvature is: 1432.6666979354738 m 1851.336986703841 m
    

      8%|██████▋                                                                        | 106/1261 [00:15<02:52,  6.68it/s]

    The radius of curvature is: 1411.850726618717 m 2719.6969232975516 m
    

      8%|██████▋                                                                        | 107/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1356.6232840885993 m 4041.7131711732177 m
    

      9%|██████▊                                                                        | 108/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1366.1441937668674 m 856.8311138807035 m
    

      9%|██████▊                                                                        | 109/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1286.3919739134558 m 945.1981966082516 m
    

      9%|██████▉                                                                        | 110/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1258.394428509198 m 994.3944855423181 m
    

      9%|██████▉                                                                        | 111/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1266.8256384346564 m 934.011250036275 m
    

      9%|███████                                                                        | 112/1261 [00:16<02:52,  6.68it/s]

    The radius of curvature is: 1184.9754302268973 m 1094.203889120428 m
    

      9%|███████                                                                        | 113/1261 [00:16<02:51,  6.68it/s]

    The radius of curvature is: 1073.0879816716356 m 2054.3486338337007 m
    

      9%|███████▏                                                                       | 114/1261 [00:17<02:51,  6.68it/s]

    The radius of curvature is: 966.401904276747 m 11298.151950312613 m
    

      9%|███████▏                                                                       | 115/1261 [00:17<02:51,  6.68it/s]

    The radius of curvature is: 933.1632298878137 m 3912.9162108559117 m
    

      9%|███████▎                                                                       | 116/1261 [00:17<02:51,  6.67it/s]

    The radius of curvature is: 955.7301149362164 m 877.4091782804417 m
    

      9%|███████▎                                                                       | 117/1261 [00:17<02:51,  6.67it/s]

    The radius of curvature is: 1118.133860169363 m 1847.3371764911487 m
    

      9%|███████▍                                                                       | 118/1261 [00:17<02:51,  6.67it/s]

    The radius of curvature is: 1019.545050413029 m 2029.771678865198 m
    

      9%|███████▍                                                                       | 119/1261 [00:17<02:51,  6.67it/s]

    The radius of curvature is: 898.7270213605137 m 1072.74317191549 m
    

     10%|███████▌                                                                       | 120/1261 [00:17<02:50,  6.67it/s]

    The radius of curvature is: 856.3539390333324 m 639.822830642496 m
    

     10%|███████▌                                                                       | 121/1261 [00:18<02:50,  6.67it/s]

    The radius of curvature is: 841.0672657555034 m 826.1786017742028 m
    

     10%|███████▋                                                                       | 122/1261 [00:18<02:50,  6.68it/s]

    The radius of curvature is: 842.5023982527273 m 775.1724849350246 m
    

     10%|███████▋                                                                       | 123/1261 [00:18<02:50,  6.68it/s]

    The radius of curvature is: 832.2819250907557 m 818.8375453651367 m
    

     10%|███████▊                                                                       | 124/1261 [00:18<02:50,  6.68it/s]

    The radius of curvature is: 727.5529427882173 m 820.81744031134 m
    

     10%|███████▊                                                                       | 125/1261 [00:18<02:50,  6.68it/s]

    The radius of curvature is: 792.4690836251209 m 5859.97787241909 m
    

     10%|███████▉                                                                       | 126/1261 [00:18<02:50,  6.67it/s]

    The radius of curvature is: 726.1562488398597 m 2331.8209084468294 m
    

     10%|███████▉                                                                       | 127/1261 [00:19<02:49,  6.68it/s]

    The radius of curvature is: 798.8309270022016 m 2289.7258800983413 m
    

     10%|████████                                                                       | 128/1261 [00:19<02:49,  6.68it/s]

    The radius of curvature is: 830.1517807337723 m 1167.808007639223 m
    

     10%|████████                                                                       | 129/1261 [00:19<02:49,  6.68it/s]

    The radius of curvature is: 870.3956188019702 m 1974.5607295111738 m
    

     10%|████████▏                                                                      | 130/1261 [00:19<02:49,  6.68it/s]

    The radius of curvature is: 966.3687206866691 m 1626.1181736819813 m
    

     10%|████████▏                                                                      | 131/1261 [00:19<02:49,  6.68it/s]

    The radius of curvature is: 1054.3388793093197 m 1197.0561231282363 m
    

     10%|████████▎                                                                      | 132/1261 [00:19<02:48,  6.68it/s]

    The radius of curvature is: 1048.308874472413 m 818.7340145549472 m
    

     11%|████████▎                                                                      | 133/1261 [00:19<02:48,  6.68it/s]

    The radius of curvature is: 1014.2973432880227 m 796.5532675102822 m
    

     11%|████████▍                                                                      | 134/1261 [00:20<02:48,  6.68it/s]

    The radius of curvature is: 951.3697288638548 m 739.6118365931835 m
    

     11%|████████▍                                                                      | 135/1261 [00:20<02:48,  6.68it/s]

    The radius of curvature is: 981.8398152683428 m 799.4652921501364 m
    

     11%|████████▌                                                                      | 136/1261 [00:20<02:48,  6.68it/s]

    The radius of curvature is: 904.9980813972674 m 1134.5033796862651 m
    

     11%|████████▌                                                                      | 137/1261 [00:20<02:48,  6.68it/s]

    The radius of curvature is: 952.2176294372971 m 1521.4224345322507 m
    

     11%|████████▋                                                                      | 138/1261 [00:20<02:48,  6.68it/s]

    The radius of curvature is: 1033.2299508644471 m 2026.809526262517 m
    

     11%|████████▋                                                                      | 139/1261 [00:20<02:47,  6.68it/s]

    The radius of curvature is: 1078.3938366162856 m 4858.76296378607 m
    

     11%|████████▊                                                                      | 140/1261 [00:20<02:47,  6.68it/s]

    The radius of curvature is: 1083.197730371962 m 63290.85329560243 m
    

     11%|████████▊                                                                      | 141/1261 [00:21<02:47,  6.68it/s]

    The radius of curvature is: 1243.0549712123423 m 1235.6536932072183 m
    

     11%|████████▉                                                                      | 142/1261 [00:21<02:47,  6.68it/s]

    The radius of curvature is: 1140.4754393462363 m 1538.5306417356987 m
    

     11%|████████▉                                                                      | 143/1261 [00:21<02:47,  6.68it/s]

    The radius of curvature is: 1144.8501015617794 m 2885.7046640890267 m
    

     11%|█████████                                                                      | 144/1261 [00:21<02:47,  6.68it/s]

    The radius of curvature is: 1180.301742400241 m 2403.030132592867 m
    

     11%|█████████                                                                      | 145/1261 [00:21<02:47,  6.68it/s]

    The radius of curvature is: 1163.3586328106041 m 1241.4198113828793 m
    

     12%|█████████▏                                                                     | 146/1261 [00:21<02:46,  6.68it/s]

    The radius of curvature is: 1101.2466289096076 m 1041.25898758353 m
    

     12%|█████████▏                                                                     | 147/1261 [00:22<02:46,  6.68it/s]

    The radius of curvature is: 1260.3771932345971 m 1034.2755861782564 m
    

     12%|█████████▎                                                                     | 148/1261 [00:22<02:46,  6.68it/s]

    The radius of curvature is: 1050.629996800429 m 1233.593040956312 m
    

     12%|█████████▎                                                                     | 149/1261 [00:22<02:46,  6.68it/s]

    The radius of curvature is: 355.3912207460166 m 2987.155469932566 m
    

     12%|█████████▍                                                                     | 150/1261 [00:22<02:46,  6.68it/s]

    The radius of curvature is: 147.17427874415975 m 9177.467186574815 m
    

     12%|█████████▍                                                                     | 151/1261 [00:22<02:46,  6.68it/s]

    The radius of curvature is: 11.69064300228254 m 3794.338835756387 m
    

     12%|█████████▌                                                                     | 152/1261 [00:22<02:45,  6.69it/s]

    The radius of curvature is: 1169.6364780609194 m 921.6056755594972 m
    

     12%|█████████▌                                                                     | 153/1261 [00:22<02:45,  6.69it/s]

    The radius of curvature is: 517.5960538814564 m 846.3869161591647 m
    

     12%|█████████▋                                                                     | 154/1261 [00:23<02:45,  6.69it/s]

    The radius of curvature is: 69.05609991773966 m 1311.1512746202852 m
    

     12%|█████████▋                                                                     | 155/1261 [00:23<02:45,  6.69it/s]

    The radius of curvature is: 1387.7526772236822 m 1537.9621541859958 m
    

     12%|█████████▊                                                                     | 156/1261 [00:23<02:45,  6.69it/s]

    The radius of curvature is: 1127.1234621452704 m 1748.7611341522447 m
    

     12%|█████████▊                                                                     | 157/1261 [00:23<02:45,  6.69it/s]

    The radius of curvature is: 1235.3813521964046 m 767.5740766913615 m
    

     13%|█████████▉                                                                     | 158/1261 [00:23<02:44,  6.69it/s]

    The radius of curvature is: 1182.4030925960506 m 653.1522906022884 m
    

     13%|█████████▉                                                                     | 159/1261 [00:23<02:44,  6.69it/s]

    The radius of curvature is: 1038.5129825311137 m 924.419129672919 m
    

     13%|██████████                                                                     | 160/1261 [00:23<02:44,  6.69it/s]

    The radius of curvature is: 1061.5838908891742 m 1474.0291062654353 m
    

     13%|██████████                                                                     | 161/1261 [00:24<02:44,  6.69it/s]

    The radius of curvature is: 996.2866311937872 m 1795.1684841673602 m
    

     13%|██████████▏                                                                    | 162/1261 [00:24<02:44,  6.69it/s]

    The radius of curvature is: 1051.8722456840133 m 2707.916907385633 m
    

     13%|██████████▏                                                                    | 163/1261 [00:24<02:44,  6.69it/s]

    The radius of curvature is: 1051.7779349897044 m 5009.226633065454 m
    

     13%|██████████▎                                                                    | 164/1261 [00:24<02:43,  6.69it/s]

    The radius of curvature is: 949.0993943383272 m 7270.024637123753 m
    

     13%|██████████▎                                                                    | 165/1261 [00:24<02:43,  6.69it/s]

    The radius of curvature is: 998.7588341950116 m 1242.3697139153387 m
    

     13%|██████████▍                                                                    | 166/1261 [00:24<02:43,  6.69it/s]

    The radius of curvature is: 812.5661161737477 m 10483.611314409918 m
    

     13%|██████████▍                                                                    | 167/1261 [00:24<02:43,  6.69it/s]

    The radius of curvature is: 1128.7088162978828 m 10177.229948324653 m
    

     13%|██████████▌                                                                    | 168/1261 [00:25<02:43,  6.69it/s]

    The radius of curvature is: 1052.8323614081805 m 16666.138454645352 m
    

     13%|██████████▌                                                                    | 169/1261 [00:25<02:43,  6.69it/s]

    The radius of curvature is: 1381.4776742093748 m 10997.692238229945 m
    

     13%|██████████▋                                                                    | 170/1261 [00:25<02:43,  6.69it/s]

    The radius of curvature is: 1371.0690927479752 m 2085.5773864649173 m
    

     14%|██████████▋                                                                    | 171/1261 [00:25<02:42,  6.69it/s]

    The radius of curvature is: 1285.9138838410283 m 1783.5616314087026 m
    

     14%|██████████▊                                                                    | 172/1261 [00:25<02:42,  6.69it/s]

    The radius of curvature is: 1381.1042431034036 m 787.4935097504518 m
    

     14%|██████████▊                                                                    | 173/1261 [00:25<02:42,  6.69it/s]

    The radius of curvature is: 1709.5821147010533 m 964.8575732626705 m
    

     14%|██████████▉                                                                    | 174/1261 [00:25<02:42,  6.70it/s]

    The radius of curvature is: 1543.9446768340856 m 1197.7416491801246 m
    

     14%|██████████▉                                                                    | 175/1261 [00:26<02:42,  6.69it/s]

    The radius of curvature is: 1657.0870436380949 m 1376.494416649881 m
    

     14%|███████████                                                                    | 176/1261 [00:26<02:42,  6.69it/s]

    The radius of curvature is: 1814.9615070723635 m 982.387290332274 m
    

     14%|███████████                                                                    | 177/1261 [00:26<02:42,  6.69it/s]

    The radius of curvature is: 1737.8860888541471 m 628.2820558408 m
    

     14%|███████████▏                                                                   | 178/1261 [00:26<02:41,  6.69it/s]

    The radius of curvature is: 1778.3848260067205 m 617.23081982674 m
    

     14%|███████████▏                                                                   | 179/1261 [00:26<02:41,  6.69it/s]

    The radius of curvature is: 1806.3425045890674 m 769.1088882064045 m
    

     14%|███████████▎                                                                   | 180/1261 [00:26<02:41,  6.69it/s]

    The radius of curvature is: 1428.7741188026364 m 929.6029571562688 m
    

     14%|███████████▎                                                                   | 181/1261 [00:27<02:41,  6.69it/s]

    The radius of curvature is: 1595.5106608974183 m 973.6886202854619 m
    

     14%|███████████▍                                                                   | 182/1261 [00:27<02:41,  6.69it/s]

    The radius of curvature is: 1280.119786119678 m 639.6626638488884 m
    

     15%|███████████▍                                                                   | 183/1261 [00:27<02:41,  6.69it/s]

    The radius of curvature is: 1109.7808547227726 m 715.9449603522916 m
    

     15%|███████████▌                                                                   | 184/1261 [00:27<02:41,  6.69it/s]

    The radius of curvature is: 929.8256513859076 m 848.2663022580695 m
    

     15%|███████████▌                                                                   | 185/1261 [00:27<02:41,  6.68it/s]

    The radius of curvature is: 789.9334893011392 m 1678.1331922698882 m
    

     15%|███████████▋                                                                   | 186/1261 [00:27<02:40,  6.68it/s]

    The radius of curvature is: 781.2190713248225 m 749.7516096065286 m
    

     15%|███████████▋                                                                   | 187/1261 [00:27<02:40,  6.68it/s]

    The radius of curvature is: 719.7809747831382 m 643.7923089503331 m
    

     15%|███████████▊                                                                   | 188/1261 [00:28<02:40,  6.68it/s]

    The radius of curvature is: 740.1779521443138 m 890.5453927241613 m
    

     15%|███████████▊                                                                   | 189/1261 [00:28<02:40,  6.68it/s]

    The radius of curvature is: 750.1881316576523 m 1674.2852870719096 m
    

     15%|███████████▉                                                                   | 190/1261 [00:28<02:40,  6.68it/s]

    The radius of curvature is: 761.3645115666853 m 1694.023809193406 m
    

     15%|███████████▉                                                                   | 191/1261 [00:28<02:40,  6.68it/s]

    The radius of curvature is: 891.0988330961898 m 1301.5424444616815 m
    

     15%|████████████                                                                   | 192/1261 [00:28<02:39,  6.68it/s]

    The radius of curvature is: 738.8234566570818 m 829.7521688751896 m
    

     15%|████████████                                                                   | 193/1261 [00:28<02:39,  6.68it/s]

    The radius of curvature is: 702.4275006842012 m 824.5721742067675 m
    

     15%|████████████▏                                                                  | 194/1261 [00:29<02:39,  6.68it/s]

    The radius of curvature is: 766.8547027255084 m 728.7553294659298 m
    

     15%|████████████▏                                                                  | 195/1261 [00:29<02:39,  6.68it/s]

    The radius of curvature is: 955.7138500040567 m 792.3534648158923 m
    

     16%|████████████▎                                                                  | 196/1261 [00:29<02:39,  6.69it/s]

    The radius of curvature is: 1060.356538165296 m 1422.1017470251797 m
    

     16%|████████████▎                                                                  | 197/1261 [00:29<02:39,  6.69it/s]

    The radius of curvature is: 1151.164854101177 m 901.9415579356256 m
    

     16%|████████████▍                                                                  | 198/1261 [00:29<02:38,  6.69it/s]

    The radius of curvature is: 1049.0164547002853 m 805.069786300547 m
    

     16%|████████████▍                                                                  | 199/1261 [00:29<02:38,  6.69it/s]

    The radius of curvature is: 1145.4352464876993 m 1272.107811975177 m
    

     16%|████████████▌                                                                  | 200/1261 [00:29<02:38,  6.69it/s]

    The radius of curvature is: 1005.4747324827865 m 1241.341743628133 m
    

     16%|████████████▌                                                                  | 201/1261 [00:30<02:38,  6.69it/s]

    The radius of curvature is: 1108.6885515814313 m 1892.410114641802 m
    

     16%|████████████▋                                                                  | 202/1261 [00:30<02:38,  6.69it/s]

    The radius of curvature is: 1199.0706708876592 m 1197.3737366241703 m
    

     16%|████████████▋                                                                  | 203/1261 [00:30<02:38,  6.69it/s]

    The radius of curvature is: 1417.3967541500665 m 8779.587723775958 m
    

     16%|████████████▊                                                                  | 204/1261 [00:30<02:38,  6.69it/s]

    The radius of curvature is: 1556.0094302506036 m 1060.1202599783683 m
    

     16%|████████████▊                                                                  | 205/1261 [00:30<02:37,  6.69it/s]

    The radius of curvature is: 1541.057659119305 m 1670.4435626507395 m
    

     16%|████████████▉                                                                  | 206/1261 [00:30<02:37,  6.69it/s]

    The radius of curvature is: 1313.2424985530613 m 891.4651119218739 m
    

     16%|████████████▉                                                                  | 207/1261 [00:30<02:37,  6.69it/s]

    The radius of curvature is: 1203.7771925578782 m 714.0835536882549 m
    

     16%|█████████████                                                                  | 208/1261 [00:31<02:37,  6.69it/s]

    The radius of curvature is: 1237.5139330117945 m 685.1709684372898 m
    

     17%|█████████████                                                                  | 209/1261 [00:31<02:37,  6.69it/s]

    The radius of curvature is: 1170.4859502384427 m 696.8810916776974 m
    

     17%|█████████████▏                                                                 | 210/1261 [00:31<02:37,  6.69it/s]

    The radius of curvature is: 1136.4690749096706 m 772.834742460336 m
    

     17%|█████████████▏                                                                 | 211/1261 [00:31<02:36,  6.69it/s]

    The radius of curvature is: 994.5938698219808 m 911.1365655083521 m
    

     17%|█████████████▎                                                                 | 212/1261 [00:31<02:36,  6.69it/s]

    The radius of curvature is: 967.5762194016352 m 1142.800711137627 m
    

     17%|█████████████▎                                                                 | 213/1261 [00:31<02:36,  6.69it/s]

    The radius of curvature is: 898.8505985403913 m 1031.3398420005722 m
    

     17%|█████████████▍                                                                 | 214/1261 [00:31<02:36,  6.69it/s]

    The radius of curvature is: 845.5715725484529 m 5318.274573219236 m
    

     17%|█████████████▍                                                                 | 215/1261 [00:32<02:36,  6.69it/s]

    The radius of curvature is: 894.3606890169989 m 799.5199180862951 m
    

     17%|█████████████▌                                                                 | 216/1261 [00:32<02:36,  6.69it/s]

    The radius of curvature is: 920.3083817951084 m 2118.0632085082984 m
    

     17%|█████████████▌                                                                 | 217/1261 [00:32<02:35,  6.69it/s]

    The radius of curvature is: 933.4516017351278 m 2810.024369228679 m
    

     17%|█████████████▋                                                                 | 218/1261 [00:32<02:35,  6.69it/s]

    The radius of curvature is: 983.6405075439537 m 1288.1878794467998 m
    

     17%|█████████████▋                                                                 | 219/1261 [00:32<02:35,  6.69it/s]

    The radius of curvature is: 1112.8474478149053 m 1056.1642033275714 m
    

     17%|█████████████▊                                                                 | 220/1261 [00:32<02:35,  6.69it/s]

    The radius of curvature is: 1061.9827477040817 m 618.6339064743767 m
    

     18%|█████████████▊                                                                 | 221/1261 [00:33<02:35,  6.69it/s]

    The radius of curvature is: 1107.4327960170717 m 856.840222446856 m
    

     18%|█████████████▉                                                                 | 222/1261 [00:33<02:35,  6.69it/s]

    The radius of curvature is: 1175.1229582209376 m 900.5434624518351 m
    

     18%|█████████████▉                                                                 | 223/1261 [00:33<02:35,  6.69it/s]

    The radius of curvature is: 1038.4222392280751 m 1970.0563236729813 m
    

     18%|██████████████                                                                 | 224/1261 [00:33<02:34,  6.69it/s]

    The radius of curvature is: 1026.9547855124404 m 1902.5313723234908 m
    

     18%|██████████████                                                                 | 225/1261 [00:33<02:34,  6.69it/s]

    The radius of curvature is: 1076.1782313262313 m 4958.944576223007 m
    

     18%|██████████████▏                                                                | 226/1261 [00:33<02:34,  6.69it/s]

    The radius of curvature is: 1002.0664421935844 m 7183.884356207163 m
    

     18%|██████████████▏                                                                | 227/1261 [00:33<02:34,  6.69it/s]

    The radius of curvature is: 1072.8981009181584 m 10634.278593141527 m
    

     18%|██████████████▎                                                                | 228/1261 [00:34<02:34,  6.69it/s]

    The radius of curvature is: 1343.5479003223925 m 2525.765232065948 m
    

     18%|██████████████▎                                                                | 229/1261 [00:34<02:34,  6.69it/s]

    The radius of curvature is: 1350.2783964714783 m 14089.740699322856 m
    

     18%|██████████████▍                                                                | 230/1261 [00:34<02:34,  6.69it/s]

    The radius of curvature is: 1489.6132764036458 m 4873.583784589851 m
    

     18%|██████████████▍                                                                | 231/1261 [00:34<02:33,  6.69it/s]

    The radius of curvature is: 1875.7937022154028 m 3347.402252393328 m
    

     18%|██████████████▌                                                                | 232/1261 [00:34<02:33,  6.69it/s]

    The radius of curvature is: 1776.8111952464324 m 690.5899194879246 m
    

     18%|██████████████▌                                                                | 233/1261 [00:34<02:33,  6.69it/s]

    The radius of curvature is: 1926.2553833824593 m 791.7133476944475 m
    

     19%|██████████████▋                                                                | 234/1261 [00:34<02:33,  6.69it/s]

    The radius of curvature is: 1940.456310526303 m 846.3953006302125 m
    

     19%|██████████████▋                                                                | 235/1261 [00:35<02:33,  6.69it/s]

    The radius of curvature is: 1771.5938654643305 m 768.6885241524278 m
    

     19%|██████████████▊                                                                | 236/1261 [00:35<02:33,  6.69it/s]

    The radius of curvature is: 1560.2122457830653 m 1021.0601829806094 m
    

     19%|██████████████▊                                                                | 237/1261 [00:35<02:33,  6.69it/s]

    The radius of curvature is: 1185.7151388864606 m 871.7332696213596 m
    

     19%|██████████████▉                                                                | 238/1261 [00:35<02:32,  6.69it/s]

    The radius of curvature is: 1325.8334008244556 m 704.460126765981 m
    

     19%|██████████████▉                                                                | 239/1261 [00:35<02:32,  6.69it/s]

    The radius of curvature is: 1476.9513713253548 m 1279.452097936637 m
    

     19%|███████████████                                                                | 240/1261 [00:35<02:32,  6.69it/s]

    The radius of curvature is: 1395.2472125919874 m 1565.2742897671355 m
    

     19%|███████████████                                                                | 241/1261 [00:36<02:32,  6.69it/s]

    The radius of curvature is: 1604.6726476499955 m 1567.5545046036846 m
    

     19%|███████████████▏                                                               | 242/1261 [00:36<02:32,  6.69it/s]

    The radius of curvature is: 1496.6543007963903 m 751.8754689465303 m
    

     19%|███████████████▏                                                               | 243/1261 [00:36<02:32,  6.69it/s]

    The radius of curvature is: 1272.8742878555086 m 809.7261273158774 m
    

     19%|███████████████▎                                                               | 244/1261 [00:36<02:32,  6.69it/s]

    The radius of curvature is: 1259.6073378751146 m 711.6947771948356 m
    

     19%|███████████████▎                                                               | 245/1261 [00:36<02:31,  6.69it/s]

    The radius of curvature is: 1161.7045038293238 m 639.8721228333623 m
    

     20%|███████████████▍                                                               | 246/1261 [00:36<02:31,  6.69it/s]

    The radius of curvature is: 1289.2297862627456 m 793.5639375602103 m
    

     20%|███████████████▍                                                               | 247/1261 [00:36<02:31,  6.69it/s]

    The radius of curvature is: 1129.9503882956035 m 972.0804328540943 m
    

     20%|███████████████▌                                                               | 248/1261 [00:37<02:31,  6.69it/s]

    The radius of curvature is: 998.9117172910941 m 1277.76632854622 m
    

     20%|███████████████▌                                                               | 249/1261 [00:37<02:31,  6.69it/s]

    The radius of curvature is: 984.3219888589302 m 1222.9145970290213 m
    

     20%|███████████████▋                                                               | 250/1261 [00:37<02:31,  6.69it/s]

    The radius of curvature is: 1016.6570214257134 m 792.1414039533183 m
    

     20%|███████████████▋                                                               | 251/1261 [00:37<02:30,  6.69it/s]

    The radius of curvature is: 1086.2801353348495 m 929.6555426792789 m
    

     20%|███████████████▊                                                               | 252/1261 [00:37<02:30,  6.69it/s]

    The radius of curvature is: 1142.5212568552797 m 1638.2354884170923 m
    

     20%|███████████████▊                                                               | 253/1261 [00:37<02:30,  6.69it/s]

    The radius of curvature is: 1255.9521259254363 m 2108.238173583325 m
    

     20%|███████████████▉                                                               | 254/1261 [00:37<02:30,  6.69it/s]

    The radius of curvature is: 1421.9254560393083 m 1679.4112961472722 m
    

     20%|███████████████▉                                                               | 255/1261 [00:38<02:30,  6.69it/s]

    The radius of curvature is: 1402.2510875892644 m 969.0300383450003 m
    

     20%|████████████████                                                               | 256/1261 [00:38<02:30,  6.69it/s]

    The radius of curvature is: 1448.708359720949 m 727.7348378122775 m
    

     20%|████████████████                                                               | 257/1261 [00:38<02:30,  6.69it/s]

    The radius of curvature is: 1524.5284710208923 m 644.195602036722 m
    

     20%|████████████████▏                                                              | 258/1261 [00:38<02:29,  6.69it/s]

    The radius of curvature is: 1615.922040938904 m 649.3278519977677 m
    

     21%|████████████████▏                                                              | 259/1261 [00:38<02:29,  6.69it/s]

    The radius of curvature is: 1513.2467262581129 m 1031.840274128202 m
    

     21%|████████████████▎                                                              | 260/1261 [00:38<02:29,  6.69it/s]

    The radius of curvature is: 1489.1927895683493 m 788.3027540973187 m
    

     21%|████████████████▎                                                              | 261/1261 [00:38<02:29,  6.69it/s]

    The radius of curvature is: 1513.972821137256 m 1346.425269443841 m
    

     21%|████████████████▍                                                              | 262/1261 [00:39<02:29,  6.69it/s]

    The radius of curvature is: 1500.7978670057048 m 810.2273234450988 m
    

     21%|████████████████▍                                                              | 263/1261 [00:39<02:29,  6.70it/s]

    The radius of curvature is: 1383.9856726450753 m 880.5809087786452 m
    

     21%|████████████████▌                                                              | 264/1261 [00:39<02:28,  6.70it/s]

    The radius of curvature is: 1603.0569439656235 m 1071.1631239624016 m
    

     21%|████████████████▌                                                              | 265/1261 [00:39<02:28,  6.70it/s]

    The radius of curvature is: 1777.5145466594531 m 1787.0406511883737 m
    

     21%|████████████████▋                                                              | 266/1261 [00:39<02:28,  6.70it/s]

    The radius of curvature is: 1608.9146864889751 m 1149.303661239307 m
    

     21%|████████████████▋                                                              | 267/1261 [00:39<02:28,  6.70it/s]

    The radius of curvature is: 1709.3367440567927 m 1150.8905363261374 m
    

     21%|████████████████▊                                                              | 268/1261 [00:40<02:28,  6.70it/s]

    The radius of curvature is: 1623.8452666761889 m 972.600084190544 m
    

     21%|████████████████▊                                                              | 269/1261 [00:40<02:28,  6.70it/s]

    The radius of curvature is: 1526.3205565750357 m 712.8545433822065 m
    

     21%|████████████████▉                                                              | 270/1261 [00:40<02:28,  6.70it/s]

    The radius of curvature is: 1460.707100809704 m 932.1779985310717 m
    

     21%|████████████████▉                                                              | 271/1261 [00:40<02:27,  6.70it/s]

    The radius of curvature is: 1387.521525595162 m 1322.3163707962901 m
    

     22%|█████████████████                                                              | 272/1261 [00:40<02:27,  6.70it/s]

    The radius of curvature is: 1326.8458791620415 m 1488.7136447304854 m
    

     22%|█████████████████                                                              | 273/1261 [00:40<02:27,  6.70it/s]

    The radius of curvature is: 1404.3173628111772 m 3352.260274933247 m
    

     22%|█████████████████▏                                                             | 274/1261 [00:40<02:27,  6.70it/s]

    The radius of curvature is: 1421.3757458639543 m 1451.5154676642505 m
    

     22%|█████████████████▏                                                             | 275/1261 [00:41<02:27,  6.70it/s]

    The radius of curvature is: 1295.9303220966003 m 1418.4582865781213 m
    

     22%|█████████████████▎                                                             | 276/1261 [00:41<02:27,  6.70it/s]

    The radius of curvature is: 1666.8862249782092 m 2638.202920825358 m
    

     22%|█████████████████▎                                                             | 277/1261 [00:41<02:26,  6.70it/s]

    The radius of curvature is: 1718.5098276982615 m 3689.034719102509 m
    

     22%|█████████████████▍                                                             | 278/1261 [00:41<02:26,  6.70it/s]

    The radius of curvature is: 1630.987002445856 m 1811.8319411926486 m
    

     22%|█████████████████▍                                                             | 279/1261 [00:41<02:26,  6.70it/s]

    The radius of curvature is: 1725.636468769322 m 896.4058345683806 m
    

     22%|█████████████████▌                                                             | 280/1261 [00:41<02:26,  6.70it/s]

    The radius of curvature is: 1639.8528012248382 m 627.0195086076802 m
    

     22%|█████████████████▌                                                             | 281/1261 [00:41<02:26,  6.70it/s]

    The radius of curvature is: 1764.0016242814856 m 986.6057279913288 m
    

     22%|█████████████████▋                                                             | 282/1261 [00:42<02:26,  6.70it/s]

    The radius of curvature is: 1773.1290253115028 m 778.1575490625597 m
    

     22%|█████████████████▋                                                             | 283/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 1894.4959121767515 m 1083.7430363840072 m
    

     23%|█████████████████▊                                                             | 284/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 1972.2088172842805 m 2218.0448421422075 m
    

     23%|█████████████████▊                                                             | 285/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 1839.7091554675615 m 4001.939464822126 m
    

     23%|█████████████████▉                                                             | 286/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 1856.8894296491626 m 7109.911179650547 m
    

     23%|█████████████████▉                                                             | 287/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 1772.699564381039 m 3034.5626281323625 m
    

     23%|██████████████████                                                             | 288/1261 [00:42<02:25,  6.70it/s]

    The radius of curvature is: 2360.1270583803903 m 1544.6236737133322 m
    

     23%|██████████████████                                                             | 289/1261 [00:43<02:25,  6.70it/s]

    The radius of curvature is: 2396.5590080608367 m 44978.2198925715 m
    

     23%|██████████████████▏                                                            | 290/1261 [00:43<02:24,  6.70it/s]

    The radius of curvature is: 2399.6993026571818 m 3677.8767127950696 m
    

     23%|██████████████████▏                                                            | 291/1261 [00:43<02:24,  6.70it/s]

    The radius of curvature is: 2259.028893795189 m 1328.6629410110415 m
    

     23%|██████████████████▎                                                            | 292/1261 [00:43<02:24,  6.70it/s]

    The radius of curvature is: 2478.9116083333374 m 1153.6708745429075 m
    

     23%|██████████████████▎                                                            | 293/1261 [00:43<02:24,  6.70it/s]

    The radius of curvature is: 2332.735784883678 m 1190.777078935685 m
    

     23%|██████████████████▍                                                            | 294/1261 [00:43<02:24,  6.70it/s]

    The radius of curvature is: 2401.9084824769493 m 1295.4615940211756 m
    

     23%|██████████████████▍                                                            | 295/1261 [00:44<02:24,  6.70it/s]

    The radius of curvature is: 2582.346577165541 m 2359.944816921331 m
    

     23%|██████████████████▌                                                            | 296/1261 [00:44<02:24,  6.70it/s]

    The radius of curvature is: 2231.909609143231 m 1703.4714985138398 m
    

     24%|██████████████████▌                                                            | 297/1261 [00:44<02:23,  6.70it/s]

    The radius of curvature is: 2456.9406662091483 m 4148.360915355514 m
    

     24%|██████████████████▋                                                            | 298/1261 [00:44<02:23,  6.70it/s]

    The radius of curvature is: 2076.627445647482 m 2589.3302287247584 m
    

     24%|██████████████████▋                                                            | 299/1261 [00:44<02:23,  6.70it/s]

    The radius of curvature is: 1862.5817765753577 m 1532.34011113277 m
    

     24%|██████████████████▊                                                            | 300/1261 [00:44<02:23,  6.70it/s]

    The radius of curvature is: 2560.4374771426574 m 1720.5047051501074 m
    

     24%|██████████████████▊                                                            | 301/1261 [00:44<02:23,  6.70it/s]

    The radius of curvature is: 2586.932532712212 m 2468.6220520554557 m
    

     24%|██████████████████▉                                                            | 302/1261 [00:45<02:23,  6.70it/s]

    The radius of curvature is: 2986.1056117657745 m 3035.018442900152 m
    

     24%|██████████████████▉                                                            | 303/1261 [00:45<02:23,  6.70it/s]

    The radius of curvature is: 3033.7809179677615 m 2692.288248424113 m
    

     24%|███████████████████                                                            | 304/1261 [00:45<02:22,  6.69it/s]

    The radius of curvature is: 2753.6330190931067 m 1029.455599210668 m
    

     24%|███████████████████                                                            | 305/1261 [00:45<02:22,  6.69it/s]

    The radius of curvature is: 3234.713270941457 m 1858.6561726908267 m
    

     24%|███████████████████▏                                                           | 306/1261 [00:45<02:22,  6.69it/s]

    The radius of curvature is: 2862.762917199842 m 1342.9856518774934 m
    

     24%|███████████████████▏                                                           | 307/1261 [00:45<02:22,  6.69it/s]

    The radius of curvature is: 2993.1494621670718 m 6528.704000478971 m
    

     24%|███████████████████▎                                                           | 308/1261 [00:46<02:22,  6.69it/s]

    The radius of curvature is: 3085.903479949882 m 12388.385738340114 m
    

     25%|███████████████████▎                                                           | 309/1261 [00:46<02:22,  6.69it/s]

    The radius of curvature is: 2653.029029729489 m 13122.799690534046 m
    

     25%|███████████████████▍                                                           | 310/1261 [00:46<02:22,  6.69it/s]

    The radius of curvature is: 2890.703404445011 m 2380.740207897289 m
    

     25%|███████████████████▍                                                           | 311/1261 [00:46<02:21,  6.69it/s]

    The radius of curvature is: 2654.7894622417266 m 7798.277745562663 m
    

     25%|███████████████████▌                                                           | 312/1261 [00:46<02:21,  6.69it/s]

    The radius of curvature is: 3330.4287077530635 m 3528.1408891529404 m
    

     25%|███████████████████▌                                                           | 313/1261 [00:46<02:21,  6.69it/s]

    The radius of curvature is: 3742.336800872517 m 5636.471176015253 m
    

     25%|███████████████████▋                                                           | 314/1261 [00:46<02:21,  6.69it/s]

    The radius of curvature is: 4082.2510832256316 m 53203.237968106405 m
    

     25%|███████████████████▋                                                           | 315/1261 [00:47<02:21,  6.69it/s]

    The radius of curvature is: 5495.22964240299 m 1817.497571415391 m
    

     25%|███████████████████▊                                                           | 316/1261 [00:47<02:21,  6.69it/s]

    The radius of curvature is: 4749.71083396481 m 1119.0671703229136 m
    

     25%|███████████████████▊                                                           | 317/1261 [00:47<02:21,  6.69it/s]

    The radius of curvature is: 7431.740392403474 m 1132.783140125045 m
    

     25%|███████████████████▉                                                           | 318/1261 [00:47<02:20,  6.69it/s]

    The radius of curvature is: 10471.829872864764 m 1131.2144247537296 m
    

     25%|███████████████████▉                                                           | 319/1261 [00:47<02:20,  6.69it/s]

    The radius of curvature is: 11646.961215274352 m 2047.118550341539 m
    

     25%|████████████████████                                                           | 320/1261 [00:47<02:20,  6.69it/s]

    The radius of curvature is: 6917.895683196685 m 2012.2322821115213 m
    

     25%|████████████████████                                                           | 321/1261 [00:47<02:20,  6.69it/s]

    The radius of curvature is: 48179.4627864763 m 1235.5819930999833 m
    

     26%|████████████████████▏                                                          | 322/1261 [00:48<02:20,  6.69it/s]

    The radius of curvature is: 38270.99670267894 m 1221.4567908756683 m
    

     26%|████████████████████▏                                                          | 323/1261 [00:48<02:20,  6.69it/s]

    The radius of curvature is: 48450.01107101933 m 1598.3362142418346 m
    

     26%|████████████████████▎                                                          | 324/1261 [00:48<02:20,  6.69it/s]

    The radius of curvature is: 12952.198673860465 m 8631.158382760446 m
    

     26%|████████████████████▎                                                          | 325/1261 [00:48<02:19,  6.69it/s]

    The radius of curvature is: 8524.12497803624 m 6288.531541805114 m
    

     26%|████████████████████▍                                                          | 326/1261 [00:48<02:19,  6.69it/s]

    The radius of curvature is: 12259.521197726723 m 6702.32708375507 m
    

     26%|████████████████████▍                                                          | 327/1261 [00:48<02:19,  6.69it/s]

    The radius of curvature is: 23868.79119190196 m 1921.5616940183268 m
    

     26%|████████████████████▌                                                          | 328/1261 [00:49<02:19,  6.69it/s]

    The radius of curvature is: 20551.19747263061 m 1157.0923162544457 m
    

     26%|████████████████████▌                                                          | 329/1261 [00:49<02:19,  6.69it/s]

    The radius of curvature is: 63776.375223282055 m 1194.151823388623 m
    

     26%|████████████████████▋                                                          | 330/1261 [00:49<02:19,  6.69it/s]

    The radius of curvature is: 78683.55485857732 m 1304.043570236416 m
    

     26%|████████████████████▋                                                          | 331/1261 [00:49<02:19,  6.69it/s]

    The radius of curvature is: 89835.85709828608 m 3115.1938760921144 m
    

     26%|████████████████████▊                                                          | 332/1261 [00:49<02:18,  6.69it/s]

    The radius of curvature is: 20659.490572933046 m 3339.9076416162407 m
    

     26%|████████████████████▊                                                          | 333/1261 [00:49<02:18,  6.69it/s]

    The radius of curvature is: 12743.376948226753 m 1160.035285570395 m
    

     26%|████████████████████▉                                                          | 334/1261 [00:49<02:18,  6.69it/s]

    The radius of curvature is: 9521.211626247143 m 3355.432024018028 m
    

     27%|████████████████████▉                                                          | 335/1261 [00:50<02:18,  6.69it/s]

    The radius of curvature is: 6307.308029792228 m 7367.313755230991 m
    

     27%|█████████████████████                                                          | 336/1261 [00:50<02:18,  6.69it/s]

    The radius of curvature is: 6688.431074696548 m 24338.147496051 m
    

     27%|█████████████████████                                                          | 337/1261 [00:50<02:18,  6.69it/s]

    The radius of curvature is: 8427.38013271349 m 3132.7179229394487 m
    

     27%|█████████████████████▏                                                         | 338/1261 [00:50<02:17,  6.69it/s]

    The radius of curvature is: 10813.25605410604 m 4443.079969213165 m
    

     27%|█████████████████████▏                                                         | 339/1261 [00:50<02:17,  6.69it/s]

    The radius of curvature is: 10378.15658261544 m 3805.8998123572574 m
    

     27%|█████████████████████▎                                                         | 340/1261 [00:50<02:17,  6.69it/s]

    The radius of curvature is: 6249.820222613338 m 1426.4464298654095 m
    

     27%|█████████████████████▎                                                         | 341/1261 [00:50<02:17,  6.69it/s]

    The radius of curvature is: 13653.217890035385 m 2170.1648725809873 m
    

     27%|█████████████████████▍                                                         | 342/1261 [00:51<02:17,  6.69it/s]

    The radius of curvature is: 21019.090201915642 m 1417.376628140601 m
    

     27%|█████████████████████▍                                                         | 343/1261 [00:51<02:17,  6.69it/s]

    The radius of curvature is: 96276.0227692917 m 2394.7423129772596 m
    

     27%|█████████████████████▌                                                         | 344/1261 [00:51<02:17,  6.69it/s]

    The radius of curvature is: 46087.04664234174 m 4494.701242895553 m
    

     27%|█████████████████████▌                                                         | 345/1261 [00:51<02:16,  6.69it/s]

    The radius of curvature is: 50125.565862601936 m 7920.5102008354415 m
    

     27%|█████████████████████▋                                                         | 346/1261 [00:51<02:16,  6.69it/s]

    The radius of curvature is: 6341.297253923911 m 1755.6570090403036 m
    

     28%|█████████████████████▋                                                         | 347/1261 [00:51<02:16,  6.69it/s]

    The radius of curvature is: 8810.25300210325 m 2196.64744466097 m
    

     28%|█████████████████████▊                                                         | 348/1261 [00:52<02:16,  6.69it/s]

    The radius of curvature is: 5432.266792612495 m 5672.536695616715 m
    

     28%|█████████████████████▊                                                         | 349/1261 [00:52<02:16,  6.69it/s]

    The radius of curvature is: 4463.9456253320395 m 8376.547124745379 m
    

     28%|█████████████████████▉                                                         | 350/1261 [00:52<02:16,  6.69it/s]

    The radius of curvature is: 4454.44804490353 m 4104.052317293691 m
    

     28%|█████████████████████▉                                                         | 351/1261 [00:52<02:16,  6.69it/s]

    The radius of curvature is: 4592.039621126928 m 61469.759400101706 m
    

     28%|██████████████████████                                                         | 352/1261 [00:52<02:15,  6.69it/s]

    The radius of curvature is: 10770.181883187957 m 5079.139354100589 m
    

     28%|██████████████████████                                                         | 353/1261 [00:52<02:15,  6.69it/s]

    The radius of curvature is: 12725.866376571168 m 1292.1579992067473 m
    

     28%|██████████████████████▏                                                        | 354/1261 [00:52<02:15,  6.69it/s]

    The radius of curvature is: 5438.007399708198 m 6314.007385283768 m
    

     28%|██████████████████████▏                                                        | 355/1261 [00:53<02:15,  6.69it/s]

    The radius of curvature is: 3477.1004949945786 m 2349.912112819636 m
    

     28%|██████████████████████▎                                                        | 356/1261 [00:53<02:15,  6.69it/s]

    The radius of curvature is: 2486.114297724432 m 2171.3564126448646 m
    

     28%|██████████████████████▎                                                        | 357/1261 [00:53<02:15,  6.69it/s]

    The radius of curvature is: 2182.5282405402586 m 3190.7133692909815 m
    

     28%|██████████████████████▍                                                        | 358/1261 [00:53<02:14,  6.69it/s]

    The radius of curvature is: 1830.5509013378007 m 19051.571912346924 m
    

     28%|██████████████████████▍                                                        | 359/1261 [00:53<02:14,  6.69it/s]

    The radius of curvature is: 1875.0206031189468 m 7479.887056725049 m
    

     29%|██████████████████████▌                                                        | 360/1261 [00:53<02:14,  6.69it/s]

    The radius of curvature is: 2425.879377339922 m 10531.732190823099 m
    

     29%|██████████████████████▌                                                        | 361/1261 [00:53<02:14,  6.69it/s]

    The radius of curvature is: 5570.221934243762 m 31836.24862740096 m
    

     29%|██████████████████████▋                                                        | 362/1261 [00:54<02:14,  6.69it/s]

    The radius of curvature is: 11380.633641805667 m 4197.3812191190955 m
    

     29%|██████████████████████▋                                                        | 363/1261 [00:54<02:14,  6.69it/s]

    The radius of curvature is: 12471.398207341597 m 3018.1140643714944 m
    

     29%|██████████████████████▊                                                        | 364/1261 [00:54<02:14,  6.69it/s]

    The radius of curvature is: 5547.67581894958 m 1923.4892799798697 m
    

     29%|██████████████████████▊                                                        | 365/1261 [00:54<02:13,  6.69it/s]

    The radius of curvature is: 3166.900235600504 m 2920.869562669906 m
    

     29%|██████████████████████▉                                                        | 366/1261 [00:54<02:13,  6.69it/s]

    The radius of curvature is: 2463.724895767135 m 3222.108792256039 m
    

     29%|██████████████████████▉                                                        | 367/1261 [00:54<02:13,  6.69it/s]

    The radius of curvature is: 1830.8846724199939 m 2122.54666391146 m
    

     29%|███████████████████████                                                        | 368/1261 [00:54<02:13,  6.69it/s]

    The radius of curvature is: 1601.5570845956815 m 1716.428240224025 m
    

     29%|███████████████████████                                                        | 369/1261 [00:55<02:13,  6.69it/s]

    The radius of curvature is: 1766.2996097467965 m 3096.67012421428 m
    

     29%|███████████████████████▏                                                       | 370/1261 [00:55<02:13,  6.69it/s]

    The radius of curvature is: 1681.9322430998955 m 5382.585453362801 m
    

     29%|███████████████████████▏                                                       | 371/1261 [00:55<02:12,  6.69it/s]

    The radius of curvature is: 1743.1743072281006 m 27428.77606818171 m
    

     30%|███████████████████████▎                                                       | 372/1261 [00:55<02:12,  6.69it/s]

    The radius of curvature is: 2362.524392665518 m 13420.601035741285 m
    

     30%|███████████████████████▎                                                       | 373/1261 [00:55<02:12,  6.69it/s]

    The radius of curvature is: 2045.8794522217313 m 2012.1653454736581 m
    

     30%|███████████████████████▍                                                       | 374/1261 [00:55<02:12,  6.69it/s]

    The radius of curvature is: 2272.609856765263 m 43390.89164085216 m
    

     30%|███████████████████████▍                                                       | 375/1261 [00:56<02:12,  6.69it/s]

    The radius of curvature is: 3294.1952657109573 m 3449.2184499966024 m
    

     30%|███████████████████████▌                                                       | 376/1261 [00:56<02:12,  6.69it/s]

    The radius of curvature is: 4614.856974461372 m 1680.294002657977 m
    

     30%|███████████████████████▌                                                       | 377/1261 [00:56<02:12,  6.69it/s]

    The radius of curvature is: 33290.5051345478 m 1457.3997417490998 m
    

     30%|███████████████████████▋                                                       | 378/1261 [00:56<02:11,  6.69it/s]

    The radius of curvature is: 294433.01883596036 m 5479.95408282855 m
    

     30%|███████████████████████▋                                                       | 379/1261 [00:56<02:11,  6.69it/s]

    The radius of curvature is: 11908.177035965113 m 6610.801864272906 m
    

     30%|███████████████████████▊                                                       | 380/1261 [00:56<02:11,  6.69it/s]

    The radius of curvature is: 6736.76253351128 m 7168.958580765187 m
    

     30%|███████████████████████▊                                                       | 381/1261 [00:56<02:11,  6.69it/s]

    The radius of curvature is: 4691.106442829897 m 1351.9562073108873 m
    

     30%|███████████████████████▉                                                       | 382/1261 [00:57<02:11,  6.69it/s]

    The radius of curvature is: 4021.304536035015 m 3516.3419545926145 m
    

     30%|███████████████████████▉                                                       | 383/1261 [00:57<02:11,  6.69it/s]

    The radius of curvature is: 3288.7250997301944 m 4402.162744364161 m
    

     30%|████████████████████████                                                       | 384/1261 [00:57<02:11,  6.69it/s]

    The radius of curvature is: 2991.536893517324 m 5565.169424352474 m
    

     31%|████████████████████████                                                       | 385/1261 [00:57<02:10,  6.69it/s]

    The radius of curvature is: 3887.4491821700162 m 2704.335358813437 m
    

     31%|████████████████████████▏                                                      | 386/1261 [00:57<02:10,  6.69it/s]

    The radius of curvature is: 5004.472208037213 m 2640.7597717262347 m
    

     31%|████████████████████████▏                                                      | 387/1261 [00:57<02:10,  6.69it/s]

    The radius of curvature is: 5735.72305754343 m 7802.650404440973 m
    

     31%|████████████████████████▎                                                      | 388/1261 [00:57<02:10,  6.69it/s]

    The radius of curvature is: 8675.4163893445 m 3067.594767606203 m
    

     31%|████████████████████████▎                                                      | 389/1261 [00:58<02:10,  6.69it/s]

    The radius of curvature is: 7314.54792523145 m 1131.4766538146557 m
    

     31%|████████████████████████▍                                                      | 390/1261 [00:58<02:10,  6.69it/s]

    The radius of curvature is: 16878.536101679365 m 1808.718121546842 m
    

     31%|████████████████████████▍                                                      | 391/1261 [00:58<02:10,  6.69it/s]

    The radius of curvature is: 29994.166238186346 m 11447.647130542111 m
    

     31%|████████████████████████▌                                                      | 392/1261 [00:58<02:09,  6.69it/s]

    The radius of curvature is: 21366.62100083497 m 8648.552244817512 m
    

     31%|████████████████████████▌                                                      | 393/1261 [00:58<02:09,  6.69it/s]

    The radius of curvature is: 10128.803575058197 m 4597.340821073921 m
    

     31%|████████████████████████▋                                                      | 394/1261 [00:58<02:09,  6.69it/s]

    The radius of curvature is: 23764.919439921287 m 27627.35111266719 m
    

     31%|████████████████████████▋                                                      | 395/1261 [00:59<02:09,  6.69it/s]

    The radius of curvature is: 7373.312864599445 m 1319.899910870809 m
    

     31%|████████████████████████▊                                                      | 396/1261 [00:59<02:09,  6.69it/s]

    The radius of curvature is: 7580.135571109087 m 6153.719956690256 m
    

     31%|████████████████████████▊                                                      | 397/1261 [00:59<02:09,  6.69it/s]

    The radius of curvature is: 6944.725626689304 m 15217.98029358129 m
    

     32%|████████████████████████▉                                                      | 398/1261 [00:59<02:08,  6.69it/s]

    The radius of curvature is: 4569.058779663974 m 7487.847978019732 m
    

     32%|████████████████████████▉                                                      | 399/1261 [00:59<02:08,  6.69it/s]

    The radius of curvature is: 4816.15137237332 m 4352.2318480068025 m
    

     32%|█████████████████████████                                                      | 400/1261 [00:59<02:08,  6.69it/s]

    The radius of curvature is: 3238.1530966426326 m 11000.69871630071 m
    

     32%|█████████████████████████                                                      | 401/1261 [00:59<02:08,  6.69it/s]

    The radius of curvature is: 3217.5498972670375 m 3307.93365073461 m
    

     32%|█████████████████████████▏                                                     | 402/1261 [01:00<02:08,  6.69it/s]

    The radius of curvature is: 3351.1924497014725 m 2325.0227970726623 m
    

     32%|█████████████████████████▏                                                     | 403/1261 [01:00<02:08,  6.69it/s]

    The radius of curvature is: 3279.1358461285704 m 7543.9565959234815 m
    

     32%|█████████████████████████▎                                                     | 404/1261 [01:00<02:08,  6.69it/s]

    The radius of curvature is: 4086.114294177848 m 2581.676019946358 m
    

     32%|█████████████████████████▎                                                     | 405/1261 [01:00<02:07,  6.69it/s]

    The radius of curvature is: 7000.098431303936 m 2816.7747309490487 m
    

     32%|█████████████████████████▍                                                     | 406/1261 [01:00<02:07,  6.69it/s]

    The radius of curvature is: 10419.79908429156 m 1239.3981537698885 m
    

     32%|█████████████████████████▍                                                     | 407/1261 [01:00<02:07,  6.69it/s]

    The radius of curvature is: 7519.786795930638 m 1592.645740638325 m
    

     32%|█████████████████████████▌                                                     | 408/1261 [01:00<02:07,  6.69it/s]

    The radius of curvature is: 502301.2247232783 m 1578.126790677601 m
    

     32%|█████████████████████████▌                                                     | 409/1261 [01:01<02:07,  6.69it/s]

    The radius of curvature is: 7497.783609138278 m 2991.6900508282465 m
    

     33%|█████████████████████████▋                                                     | 410/1261 [01:01<02:07,  6.69it/s]

    The radius of curvature is: 5015.127365549254 m 26343.693817279596 m
    

     33%|█████████████████████████▋                                                     | 411/1261 [01:01<02:07,  6.69it/s]

    The radius of curvature is: 5392.782430209791 m 2424.1451516674974 m
    

     33%|█████████████████████████▊                                                     | 412/1261 [01:01<02:06,  6.69it/s]

    The radius of curvature is: 6491.845503711857 m 5551.788813807608 m
    

     33%|█████████████████████████▊                                                     | 413/1261 [01:01<02:06,  6.69it/s]

    The radius of curvature is: 7505.633010681379 m 2676.6871357368486 m
    

     33%|█████████████████████████▉                                                     | 414/1261 [01:01<02:06,  6.69it/s]

    The radius of curvature is: 21504.037278583277 m 2460.8380345127093 m
    

     33%|█████████████████████████▉                                                     | 415/1261 [01:02<02:06,  6.69it/s]

    The radius of curvature is: 20218.583079203083 m 2315.877889675926 m
    

     33%|██████████████████████████                                                     | 416/1261 [01:02<02:06,  6.69it/s]

    The radius of curvature is: 12229.56361252952 m 7273.075323083233 m
    

     33%|██████████████████████████                                                     | 417/1261 [01:02<02:06,  6.69it/s]

    The radius of curvature is: 5711.168709827079 m 2576.3870050085684 m
    

     33%|██████████████████████████▏                                                    | 418/1261 [01:02<02:05,  6.69it/s]

    The radius of curvature is: 4781.7039596547065 m 1296.5385685147387 m
    

     33%|██████████████████████████▏                                                    | 419/1261 [01:02<02:05,  6.69it/s]

    The radius of curvature is: 4569.442213398006 m 1490.0498228867084 m
    

     33%|██████████████████████████▎                                                    | 420/1261 [01:02<02:05,  6.69it/s]

    The radius of curvature is: 4380.709726919716 m 1467.596667792788 m
    

     33%|██████████████████████████▍                                                    | 421/1261 [01:02<02:05,  6.69it/s]

    The radius of curvature is: 4885.691519260825 m 8358.82933473266 m
    

     33%|██████████████████████████▍                                                    | 422/1261 [01:03<02:05,  6.69it/s]

    The radius of curvature is: 5051.666450759021 m 3248.278463700416 m
    

     34%|██████████████████████████▌                                                    | 423/1261 [01:03<02:05,  6.69it/s]

    The radius of curvature is: 4357.6633669834855 m 6931.961455234563 m
    

     34%|██████████████████████████▌                                                    | 424/1261 [01:03<02:05,  6.69it/s]

    The radius of curvature is: 4503.133849639305 m 1835.4322986302109 m
    

     34%|██████████████████████████▋                                                    | 425/1261 [01:03<02:04,  6.69it/s]

    The radius of curvature is: 6948.7601029607795 m 1487.1780733245894 m
    

     34%|██████████████████████████▋                                                    | 426/1261 [01:03<02:04,  6.69it/s]

    The radius of curvature is: 6283.538579296447 m 2946.6784228558563 m
    

     34%|██████████████████████████▊                                                    | 427/1261 [01:03<02:04,  6.69it/s]

    The radius of curvature is: 7196.808913146718 m 3541.845057238749 m
    

     34%|██████████████████████████▊                                                    | 428/1261 [01:03<02:04,  6.69it/s]

    The radius of curvature is: 8155.646790711653 m 4061.571479566797 m
    

     34%|██████████████████████████▉                                                    | 429/1261 [01:04<02:04,  6.69it/s]

    The radius of curvature is: 8452.330278563104 m 4456.205829105234 m
    

     34%|██████████████████████████▉                                                    | 430/1261 [01:04<02:04,  6.69it/s]

    The radius of curvature is: 8540.02549844884 m 3226.992696657419 m
    

     34%|███████████████████████████                                                    | 431/1261 [01:04<02:04,  6.69it/s]

    The radius of curvature is: 6812.413595363343 m 3191.9446862824457 m
    

     34%|███████████████████████████                                                    | 432/1261 [01:04<02:03,  6.69it/s]

    The radius of curvature is: 6638.714153692248 m 2756.723097577002 m
    

     34%|███████████████████████████▏                                                   | 433/1261 [01:04<02:03,  6.69it/s]

    The radius of curvature is: 94646.18600592497 m 73360.45872357106 m
    

     34%|███████████████████████████▏                                                   | 434/1261 [01:04<02:03,  6.69it/s]

    The radius of curvature is: 17217.776624172904 m 6671.040828622799 m
    

     34%|███████████████████████████▎                                                   | 435/1261 [01:04<02:03,  6.69it/s]

    The radius of curvature is: 42618.7062748637 m 3638.0716207352684 m
    

     35%|███████████████████████████▎                                                   | 436/1261 [01:05<02:03,  6.69it/s]

    The radius of curvature is: 17616.69584811746 m 1895.3337357947657 m
    

     35%|███████████████████████████▍                                                   | 437/1261 [01:05<02:03,  6.69it/s]

    The radius of curvature is: 207349.5847995361 m 1845.1175476219419 m
    

     35%|███████████████████████████▍                                                   | 438/1261 [01:05<02:02,  6.69it/s]

    The radius of curvature is: 8414.883594365805 m 1642.1976497486885 m
    

     35%|███████████████████████████▌                                                   | 439/1261 [01:05<02:02,  6.69it/s]

    The radius of curvature is: 8703.33460605616 m 3077.8132412474147 m
    

     35%|███████████████████████████▌                                                   | 440/1261 [01:05<02:02,  6.69it/s]

    The radius of curvature is: 10574.813776129846 m 12951.265404333495 m
    

     35%|███████████████████████████▋                                                   | 441/1261 [01:05<02:02,  6.69it/s]

    The radius of curvature is: 7518.289920681509 m 4882.581241853784 m
    

     35%|███████████████████████████▋                                                   | 442/1261 [01:06<02:02,  6.69it/s]

    The radius of curvature is: 11632.909961052 m 3383.785370503235 m
    

     35%|███████████████████████████▊                                                   | 443/1261 [01:06<02:02,  6.69it/s]

    The radius of curvature is: 8572.150215092905 m 5964.194471974001 m
    

     35%|███████████████████████████▊                                                   | 444/1261 [01:06<02:02,  6.69it/s]

    The radius of curvature is: 5731.862405911184 m 2500.2318300612574 m
    

     35%|███████████████████████████▉                                                   | 445/1261 [01:06<02:01,  6.69it/s]

    The radius of curvature is: 3329.504527994484 m 6314.776972722165 m
    

     35%|███████████████████████████▉                                                   | 446/1261 [01:06<02:01,  6.69it/s]

    The radius of curvature is: 3723.8971704449077 m 25565.774741056175 m
    

     35%|████████████████████████████                                                   | 447/1261 [01:06<02:01,  6.69it/s]

    The radius of curvature is: 3473.8563832830673 m 6713.304365212912 m
    

     36%|████████████████████████████                                                   | 448/1261 [01:06<02:01,  6.69it/s]

    The radius of curvature is: 5915.125170487638 m 1281.9243893061696 m
    

     36%|████████████████████████████▏                                                  | 449/1261 [01:07<02:01,  6.69it/s]

    The radius of curvature is: 14520.012293379892 m 1192.5188218114622 m
    

     36%|████████████████████████████▏                                                  | 450/1261 [01:07<02:01,  6.69it/s]

    The radius of curvature is: 23635.72516429155 m 1859.344779221299 m
    

     36%|████████████████████████████▎                                                  | 451/1261 [01:07<02:00,  6.69it/s]

    The radius of curvature is: 52673.110888069146 m 2908.9445078699787 m
    

     36%|████████████████████████████▎                                                  | 452/1261 [01:07<02:00,  6.69it/s]

    The radius of curvature is: 96010.63828850297 m 8227.195917780584 m
    

     36%|████████████████████████████▍                                                  | 453/1261 [01:07<02:00,  6.69it/s]

    The radius of curvature is: 19296.151715468535 m 4610.3066618924095 m
    

     36%|████████████████████████████▍                                                  | 454/1261 [01:07<02:00,  6.69it/s]

    The radius of curvature is: 76751.34817414047 m 3219.77353881645 m
    

     36%|████████████████████████████▌                                                  | 455/1261 [01:07<02:00,  6.69it/s]

    The radius of curvature is: 46849.64836428367 m 2661.480573842606 m
    

     36%|████████████████████████████▌                                                  | 456/1261 [01:08<02:00,  6.69it/s]

    The radius of curvature is: 54927.11492383587 m 4507.229232630815 m
    

     36%|████████████████████████████▋                                                  | 457/1261 [01:08<02:00,  6.69it/s]

    The radius of curvature is: 4253.943938991012 m 22490.85121759711 m
    

     36%|████████████████████████████▋                                                  | 458/1261 [01:08<01:59,  6.69it/s]

    The radius of curvature is: 7379.9127783679105 m 212590.13876340375 m
    

     36%|████████████████████████████▊                                                  | 459/1261 [01:08<01:59,  6.70it/s]

    The radius of curvature is: 12263.566445324881 m 4074.344608561265 m
    

     36%|████████████████████████████▊                                                  | 460/1261 [01:08<01:59,  6.70it/s]

    The radius of curvature is: 25015.463591879427 m 1926.032478395267 m
    

     37%|████████████████████████████▉                                                  | 461/1261 [01:08<01:59,  6.70it/s]

    The radius of curvature is: 24281.576499180537 m 1701.2061834205706 m
    

     37%|████████████████████████████▉                                                  | 462/1261 [01:09<01:59,  6.70it/s]

    The radius of curvature is: 14817.481543529118 m 2798.269668465857 m
    

     37%|█████████████████████████████                                                  | 463/1261 [01:09<01:59,  6.70it/s]

    The radius of curvature is: 10739.791882034238 m 5488.148311429297 m
    

     37%|█████████████████████████████                                                  | 464/1261 [01:09<01:59,  6.70it/s]

    The radius of curvature is: 5681.081793978258 m 10364.775143388266 m
    

     37%|█████████████████████████████▏                                                 | 465/1261 [01:09<01:58,  6.70it/s]

    The radius of curvature is: 3626.850312723108 m 1876.1768407378995 m
    

     37%|█████████████████████████████▏                                                 | 466/1261 [01:09<01:58,  6.70it/s]

    The radius of curvature is: 3720.5583770871067 m 1037.1899992156116 m
    

     37%|█████████████████████████████▎                                                 | 467/1261 [01:09<01:58,  6.70it/s]

    The radius of curvature is: 3142.403372956838 m 2269.414316302883 m
    

     37%|█████████████████████████████▎                                                 | 468/1261 [01:09<01:58,  6.70it/s]

    The radius of curvature is: 3012.0517073237743 m 20207.006281832917 m
    

     37%|█████████████████████████████▍                                                 | 469/1261 [01:10<01:58,  6.70it/s]

    The radius of curvature is: 3176.1393194296184 m 4567.352908329885 m
    

     37%|█████████████████████████████▍                                                 | 470/1261 [01:10<01:58,  6.70it/s]

    The radius of curvature is: 3248.8340616482838 m 2815.9796218283905 m
    

     37%|█████████████████████████████▌                                                 | 471/1261 [01:10<01:57,  6.70it/s]

    The radius of curvature is: 3944.499166829284 m 4304.305740743516 m
    

     37%|█████████████████████████████▌                                                 | 472/1261 [01:10<01:57,  6.69it/s]

    The radius of curvature is: 3994.1010946830297 m 29945.676308904924 m
    

     38%|█████████████████████████████▋                                                 | 473/1261 [01:10<01:57,  6.69it/s]

    The radius of curvature is: 3510.2936763140383 m 1584.4481616232456 m
    

     38%|█████████████████████████████▋                                                 | 474/1261 [01:10<01:57,  6.69it/s]

    The radius of curvature is: 3844.5119185510803 m 1341.1103271038533 m
    

     38%|█████████████████████████████▊                                                 | 475/1261 [01:10<01:57,  6.69it/s]

    The radius of curvature is: 4748.495020596225 m 24561.071634400454 m
    

     38%|█████████████████████████████▊                                                 | 476/1261 [01:11<01:57,  6.69it/s]

    The radius of curvature is: 10840.864143834839 m 50242.63281998189 m
    

     38%|█████████████████████████████▉                                                 | 477/1261 [01:11<01:57,  6.69it/s]

    The radius of curvature is: 35179.5390860117 m 1257.6208529877392 m
    

     38%|█████████████████████████████▉                                                 | 478/1261 [01:11<01:56,  6.69it/s]

    The radius of curvature is: 54468.05329778245 m 3412.191973006444 m
    

     38%|██████████████████████████████                                                 | 479/1261 [01:11<01:56,  6.69it/s]

    The radius of curvature is: 1307061.336169675 m 1376.5211227544146 m
    

     38%|██████████████████████████████                                                 | 480/1261 [01:11<01:56,  6.69it/s]

    The radius of curvature is: 6210.792005199704 m 1496.7526214656887 m
    

     38%|██████████████████████████████▏                                                | 481/1261 [01:11<01:56,  6.69it/s]

    The radius of curvature is: 6854.899147172088 m 3889.276976538596 m
    

     38%|██████████████████████████████▏                                                | 482/1261 [01:12<01:56,  6.69it/s]

    The radius of curvature is: 4214.140078404082 m 20220.804376795244 m
    

     38%|██████████████████████████████▎                                                | 483/1261 [01:12<01:56,  6.69it/s]

    The radius of curvature is: 6075.714621316696 m 3276.040188724549 m
    

     38%|██████████████████████████████▎                                                | 484/1261 [01:12<01:56,  6.69it/s]

    The radius of curvature is: 9206.96697470209 m 53418.329751429505 m
    

     38%|██████████████████████████████▍                                                | 485/1261 [01:12<01:55,  6.69it/s]

    The radius of curvature is: 21659.713545326857 m 1868.2237752343071 m
    

     39%|██████████████████████████████▍                                                | 486/1261 [01:12<01:55,  6.69it/s]

    The radius of curvature is: 782631.390356325 m 3567.249178868257 m
    

     39%|██████████████████████████████▌                                                | 487/1261 [01:12<01:55,  6.69it/s]

    The radius of curvature is: 99745.56538328447 m 3297.8899547901574 m
    

     39%|██████████████████████████████▌                                                | 488/1261 [01:12<01:55,  6.69it/s]

    The radius of curvature is: 18737.685485951843 m 54201.262149040274 m
    

     39%|██████████████████████████████▋                                                | 489/1261 [01:13<01:55,  6.69it/s]

    The radius of curvature is: 8344.354980664419 m 1753.8336273594614 m
    

     39%|██████████████████████████████▋                                                | 490/1261 [01:13<01:55,  6.69it/s]

    The radius of curvature is: 12245.226633328826 m 7735.349502180562 m
    

     39%|██████████████████████████████▊                                                | 491/1261 [01:13<01:55,  6.69it/s]

    The radius of curvature is: 9399.534727756092 m 4369.848049034927 m
    

     39%|██████████████████████████████▊                                                | 492/1261 [01:13<01:54,  6.69it/s]

    The radius of curvature is: 49672.174123419456 m 18605.44383663986 m
    

     39%|██████████████████████████████▉                                                | 493/1261 [01:13<01:54,  6.69it/s]

    The radius of curvature is: 28647.695652480543 m 9840.38708481114 m
    

     39%|██████████████████████████████▉                                                | 494/1261 [01:13<01:54,  6.69it/s]

    The radius of curvature is: 24544.291678755875 m 4654.015768061632 m
    

     39%|███████████████████████████████                                                | 495/1261 [01:13<01:54,  6.69it/s]

    The radius of curvature is: 10335.645110388878 m 8156.841969610013 m
    

     39%|███████████████████████████████                                                | 496/1261 [01:14<01:54,  6.69it/s]

    The radius of curvature is: 18555.24358444985 m 3877.1814660551595 m
    

     39%|███████████████████████████████▏                                               | 497/1261 [01:14<01:54,  6.69it/s]

    The radius of curvature is: 65633.93405399311 m 1249.8104857414733 m
    

     39%|███████████████████████████████▏                                               | 498/1261 [01:14<01:53,  6.69it/s]

    The radius of curvature is: 270429.4932317032 m 2411.00424814866 m
    

     40%|███████████████████████████████▎                                               | 499/1261 [01:14<01:53,  6.69it/s]

    The radius of curvature is: 52555.627438652526 m 2605.699364077798 m
    

     40%|███████████████████████████████▎                                               | 500/1261 [01:14<01:53,  6.69it/s]

    The radius of curvature is: 10504.157170018916 m 2737.117739544332 m
    

     40%|███████████████████████████████▍                                               | 501/1261 [01:14<01:53,  6.69it/s]

    The radius of curvature is: 9402.153557778685 m 5511.853785304399 m
    

     40%|███████████████████████████████▍                                               | 502/1261 [01:15<01:53,  6.69it/s]

    The radius of curvature is: 30249.938607746815 m 11419.153213886979 m
    

     40%|███████████████████████████████▌                                               | 503/1261 [01:15<01:53,  6.69it/s]

    The radius of curvature is: 15915.930721750867 m 1796.0910150979335 m
    

     40%|███████████████████████████████▌                                               | 504/1261 [01:15<01:53,  6.69it/s]

    The radius of curvature is: 53717.49885055324 m 2045.02638254767 m
    

     40%|███████████████████████████████▋                                               | 505/1261 [01:15<01:52,  6.69it/s]

    The radius of curvature is: 72298.8083282919 m 23868.088261446028 m
    

     40%|███████████████████████████████▋                                               | 506/1261 [01:15<01:52,  6.69it/s]

    The radius of curvature is: 13766.969930401705 m 98936.42784411716 m
    

     40%|███████████████████████████████▊                                               | 507/1261 [01:15<01:52,  6.69it/s]

    The radius of curvature is: 6625.410308510674 m 2877.255141228391 m
    

     40%|███████████████████████████████▊                                               | 508/1261 [01:15<01:52,  6.69it/s]

    The radius of curvature is: 8974.893581243741 m 1994.7561607272141 m
    

     40%|███████████████████████████████▉                                               | 509/1261 [01:16<01:52,  6.69it/s]

    The radius of curvature is: 16770.534863721234 m 3401.412656213834 m
    

     40%|███████████████████████████████▉                                               | 510/1261 [01:16<01:52,  6.69it/s]

    The radius of curvature is: 4008.460125118421 m 2692.8170165530164 m
    

     41%|████████████████████████████████                                               | 511/1261 [01:16<01:52,  6.69it/s]

    The radius of curvature is: 4898.286813435625 m 23907.244073680446 m
    

     41%|████████████████████████████████                                               | 512/1261 [01:16<01:51,  6.69it/s]

    The radius of curvature is: 10942.745425848694 m 16049.51970896791 m
    

     41%|████████████████████████████████▏                                              | 513/1261 [01:16<01:51,  6.69it/s]

    The radius of curvature is: 6028.808787714127 m 6045.0504327708695 m
    

     41%|████████████████████████████████▏                                              | 514/1261 [01:16<01:51,  6.69it/s]

    The radius of curvature is: 9241.6636378071 m 2374.014916570376 m
    

     41%|████████████████████████████████▎                                              | 515/1261 [01:16<01:51,  6.69it/s]

    The radius of curvature is: 31496.2617380409 m 2038.3100273007442 m
    

     41%|████████████████████████████████▎                                              | 516/1261 [01:17<01:51,  6.69it/s]

    The radius of curvature is: 48977.55765273871 m 4049.774454928624 m
    

     41%|████████████████████████████████▍                                              | 517/1261 [01:17<01:51,  6.69it/s]

    The radius of curvature is: 30433.099580394628 m 4223.926352121922 m
    

     41%|████████████████████████████████▍                                              | 518/1261 [01:17<01:51,  6.69it/s]

    The radius of curvature is: 50442.679329140345 m 45458.65893475584 m
    

     41%|████████████████████████████████▌                                              | 519/1261 [01:17<01:50,  6.69it/s]

    The radius of curvature is: 84870.04196895282 m 1687.340385587633 m
    

     41%|████████████████████████████████▌                                              | 520/1261 [01:17<01:50,  6.69it/s]

    The radius of curvature is: 36040.111217014084 m 2065.3453064282476 m
    

     41%|████████████████████████████████▋                                              | 521/1261 [01:17<01:50,  6.69it/s]

    The radius of curvature is: 8366.333523322217 m 2200.5905349247346 m
    

     41%|████████████████████████████████▋                                              | 522/1261 [01:18<01:50,  6.69it/s]

    The radius of curvature is: 5602.926692554986 m 1675.9943054334099 m
    

     41%|████████████████████████████████▊                                              | 523/1261 [01:18<01:50,  6.69it/s]

    The radius of curvature is: 8947.904445147273 m 2389.3142674695973 m
    

     42%|████████████████████████████████▊                                              | 524/1261 [01:18<01:50,  6.69it/s]

    The radius of curvature is: 64913.18394431403 m 3548.048031826305 m
    

     42%|████████████████████████████████▉                                              | 525/1261 [01:18<01:50,  6.69it/s]

    The radius of curvature is: 16442.531515402756 m 4391.3291090921975 m
    

     42%|████████████████████████████████▉                                              | 526/1261 [01:18<01:49,  6.69it/s]

    The radius of curvature is: 16654.711493798794 m 17203.502108774668 m
    

     42%|█████████████████████████████████                                              | 527/1261 [01:18<01:49,  6.69it/s]

    The radius of curvature is: 12919.50156304373 m 2750.6001692316386 m
    

     42%|█████████████████████████████████                                              | 528/1261 [01:18<01:49,  6.69it/s]

    The radius of curvature is: 8346.068491336962 m 3369.7536941841736 m
    

     42%|█████████████████████████████████▏                                             | 529/1261 [01:19<01:49,  6.69it/s]

    The radius of curvature is: 41089.82483913739 m 1704.6840407931286 m
    

     42%|█████████████████████████████████▏                                             | 530/1261 [01:19<01:49,  6.69it/s]

    The radius of curvature is: 14288.851404402902 m 3465.009324494128 m
    

     42%|█████████████████████████████████▎                                             | 531/1261 [01:19<01:49,  6.69it/s]

    The radius of curvature is: 277660.75520763325 m 2233.0543956978186 m
    

     42%|█████████████████████████████████▎                                             | 532/1261 [01:19<01:49,  6.69it/s]

    The radius of curvature is: 24254.554834868082 m 2832.4988637604843 m
    

     42%|█████████████████████████████████▍                                             | 533/1261 [01:19<01:48,  6.69it/s]

    The radius of curvature is: 7140.571827244467 m 26967.538742531866 m
    

     42%|█████████████████████████████████▍                                             | 534/1261 [01:19<01:48,  6.69it/s]

    The radius of curvature is: 3907.4074447449016 m 18151.701620253487 m
    

     42%|█████████████████████████████████▌                                             | 535/1261 [01:20<01:48,  6.69it/s]

    The radius of curvature is: 2450.8091234450576 m 3836.5263788174107 m
    

     43%|█████████████████████████████████▌                                             | 536/1261 [01:20<01:48,  6.69it/s]

    The radius of curvature is: 2107.745872225132 m 1505.1246646388122 m
    

     43%|█████████████████████████████████▋                                             | 537/1261 [01:20<01:48,  6.68it/s]

    The radius of curvature is: 1907.9163733994674 m 680.6632724892173 m
    

     43%|█████████████████████████████████▋                                             | 538/1261 [01:20<01:48,  6.68it/s]

    The radius of curvature is: 2001.6249079513807 m 219.08625754487684 m
    

     43%|█████████████████████████████████▊                                             | 539/1261 [01:20<01:48,  6.68it/s]

    The radius of curvature is: 1173.1301365549903 m 289.8607340670717 m
    

     43%|█████████████████████████████████▊                                             | 540/1261 [01:20<01:47,  6.68it/s]

    The radius of curvature is: 3214.178158309149 m 165.57335535659797 m
    

     43%|█████████████████████████████████▉                                             | 541/1261 [01:20<01:47,  6.68it/s]

    The radius of curvature is: 685.5879048369954 m 1188.234576889196 m
    

     43%|█████████████████████████████████▉                                             | 542/1261 [01:21<01:47,  6.68it/s]

    The radius of curvature is: 607.8069246703882 m 64827.209126394475 m
    

     43%|██████████████████████████████████                                             | 543/1261 [01:21<01:47,  6.68it/s]

    The radius of curvature is: 707.1793978617286 m 1407.1610396528508 m
    

     43%|██████████████████████████████████                                             | 544/1261 [01:21<01:47,  6.68it/s]

    The radius of curvature is: 1154.7479233411493 m 498.4403049042036 m
    

     43%|██████████████████████████████████▏                                            | 545/1261 [01:21<01:47,  6.68it/s]

    The radius of curvature is: 638.3806577654311 m 277.80825679281406 m
    

     43%|██████████████████████████████████▏                                            | 546/1261 [01:21<01:46,  6.68it/s]

    The radius of curvature is: 971.5632402231043 m 199.000257519929 m
    

     43%|██████████████████████████████████▎                                            | 547/1261 [01:21<01:46,  6.68it/s]

    The radius of curvature is: 569.1515707431539 m 192.64905015642825 m
    

     43%|██████████████████████████████████▎                                            | 548/1261 [01:21<01:46,  6.68it/s]

    The radius of curvature is: 2749.494175183422 m 107.73794037086817 m
    

     44%|██████████████████████████████████▍                                            | 549/1261 [01:22<01:46,  6.68it/s]

    The radius of curvature is: 1094.4745950184079 m 116.7892130492115 m
    

     44%|██████████████████████████████████▍                                            | 550/1261 [01:22<01:46,  6.68it/s]

    The radius of curvature is: 734.5755889876745 m 121.04799607466146 m
    

     44%|██████████████████████████████████▌                                            | 551/1261 [01:22<01:46,  6.68it/s]

    The radius of curvature is: 500.9169149635716 m 511.90798028155274 m
    

     44%|██████████████████████████████████▌                                            | 552/1261 [01:22<01:46,  6.68it/s]

    The radius of curvature is: 329.5456122720605 m 543.3046024228229 m
    

     44%|██████████████████████████████████▋                                            | 553/1261 [01:22<01:45,  6.68it/s]

    The radius of curvature is: 333.43526292343586 m 287.8723246572233 m
    

     44%|██████████████████████████████████▋                                            | 554/1261 [01:22<01:45,  6.68it/s]

    The radius of curvature is: 209.03004078992964 m 583.7948278604687 m
    

     44%|██████████████████████████████████▊                                            | 555/1261 [01:23<01:45,  6.69it/s]

    The radius of curvature is: 171.30409684652835 m 420.7391573598085 m
    

     44%|██████████████████████████████████▊                                            | 556/1261 [01:23<01:45,  6.68it/s]

    The radius of curvature is: 283.64106202496174 m 520.0659182854482 m
    

     44%|██████████████████████████████████▉                                            | 557/1261 [01:23<01:45,  6.68it/s]

    The radius of curvature is: 191.63637506062335 m 421.53057385435693 m
    

     44%|██████████████████████████████████▉                                            | 558/1261 [01:23<01:45,  6.69it/s]

    The radius of curvature is: 217.6460343346973 m 660.3774499158831 m
    

     44%|███████████████████████████████████                                            | 559/1261 [01:23<01:45,  6.69it/s]

    The radius of curvature is: 131.51510756049902 m 583.2859057004652 m
    

     44%|███████████████████████████████████                                            | 560/1261 [01:23<01:44,  6.69it/s]

    The radius of curvature is: 241.19102975902612 m 625.5737867760947 m
    

     44%|███████████████████████████████████▏                                           | 561/1261 [01:23<01:44,  6.69it/s]

    The radius of curvature is: 240.42885032230674 m 1875.7949998274084 m
    

     45%|███████████████████████████████████▏                                           | 562/1261 [01:24<01:44,  6.69it/s]

    The radius of curvature is: 240.77018711143242 m 3424.8786115982903 m
    

     45%|███████████████████████████████████▎                                           | 563/1261 [01:24<01:44,  6.69it/s]

    The radius of curvature is: 78.76761288224263 m 1399.2587695879986 m
    

     45%|███████████████████████████████████▎                                           | 564/1261 [01:24<01:44,  6.69it/s]

    The radius of curvature is: 350.65856061916384 m 454.4287104211624 m
    

     45%|███████████████████████████████████▍                                           | 565/1261 [01:24<01:44,  6.69it/s]

    The radius of curvature is: 164.37411669713748 m 2190.7181377716547 m
    

     45%|███████████████████████████████████▍                                           | 566/1261 [01:24<01:43,  6.69it/s]

    The radius of curvature is: 447.3769037103873 m 9841.664365045992 m
    

     45%|███████████████████████████████████▌                                           | 567/1261 [01:24<01:43,  6.69it/s]

    The radius of curvature is: 652.278343260089 m 29850.001174619865 m
    

     45%|███████████████████████████████████▌                                           | 568/1261 [01:24<01:43,  6.69it/s]

    The radius of curvature is: 190.6416517589282 m 2132.0440156451186 m
    

     45%|███████████████████████████████████▋                                           | 569/1261 [01:25<01:43,  6.69it/s]

    The radius of curvature is: 136.87671591718174 m 1348.6828859964676 m
    

     45%|███████████████████████████████████▋                                           | 570/1261 [01:25<01:43,  6.69it/s]

    The radius of curvature is: 280.90912732237416 m 4149.1504426600295 m
    

     45%|███████████████████████████████████▊                                           | 571/1261 [01:25<01:43,  6.69it/s]

    The radius of curvature is: 3201.1070136544413 m 3134.5123317825705 m
    

     45%|███████████████████████████████████▊                                           | 572/1261 [01:25<01:43,  6.69it/s]

    The radius of curvature is: 3935.953757887216 m 1473.008609731698 m
    

     45%|███████████████████████████████████▉                                           | 573/1261 [01:25<01:42,  6.69it/s]

    The radius of curvature is: 352.4865855331849 m 1251.805782001283 m
    

     46%|███████████████████████████████████▉                                           | 574/1261 [01:25<01:42,  6.69it/s]

    The radius of curvature is: 89.51647994731336 m 921.4691270479476 m
    

     46%|████████████████████████████████████                                           | 575/1261 [01:25<01:42,  6.69it/s]

    The radius of curvature is: 189.63788087072388 m 798.3409371255556 m
    

     46%|████████████████████████████████████                                           | 576/1261 [01:26<01:42,  6.69it/s]

    The radius of curvature is: 833.5091827906508 m 1141.3638286213773 m
    

     46%|████████████████████████████████████▏                                          | 577/1261 [01:26<01:42,  6.69it/s]

    The radius of curvature is: 116.58793871508699 m 4034.9762154278874 m
    

     46%|████████████████████████████████████▏                                          | 578/1261 [01:26<01:42,  6.69it/s]

    The radius of curvature is: 112.69902168949973 m 3909.142339578592 m
    

     46%|████████████████████████████████████▎                                          | 579/1261 [01:26<01:41,  6.69it/s]

    The radius of curvature is: 177.3886822233956 m 4039.384811839807 m
    

     46%|████████████████████████████████████▎                                          | 580/1261 [01:26<01:41,  6.69it/s]

    The radius of curvature is: 78.63819018985863 m 10075.695275740653 m
    

     46%|████████████████████████████████████▍                                          | 581/1261 [01:26<01:41,  6.69it/s]

    The radius of curvature is: 83.03841011453655 m 3087.099691865703 m
    

     46%|████████████████████████████████████▍                                          | 582/1261 [01:27<01:41,  6.69it/s]

    The radius of curvature is: 85.7058211193861 m 3522.618239770122 m
    

     46%|████████████████████████████████████▌                                          | 583/1261 [01:27<01:41,  6.69it/s]

    The radius of curvature is: 60.1097303942682 m 2884.2146073722047 m
    

     46%|████████████████████████████████████▌                                          | 584/1261 [01:27<01:41,  6.69it/s]

    The radius of curvature is: 61.98717214775596 m 21401.949338283932 m
    

     46%|████████████████████████████████████▋                                          | 585/1261 [01:27<01:41,  6.69it/s]

    The radius of curvature is: 106.92269177828003 m 5884.250149593527 m
    

     46%|████████████████████████████████████▋                                          | 586/1261 [01:27<01:40,  6.69it/s]

    The radius of curvature is: 96.3252139097968 m 3480.034999789766 m
    

     47%|████████████████████████████████████▊                                          | 587/1261 [01:27<01:40,  6.69it/s]

    The radius of curvature is: 41.01766602913206 m 908.3018378303475 m
    

     47%|████████████████████████████████████▊                                          | 588/1261 [01:27<01:40,  6.69it/s]

    The radius of curvature is: 45.23371313190907 m 724.5460660699345 m
    

     47%|████████████████████████████████████▉                                          | 589/1261 [01:28<01:40,  6.69it/s]

    The radius of curvature is: 88.92874194929189 m 1291.7104174951533 m
    

     47%|████████████████████████████████████▉                                          | 590/1261 [01:28<01:40,  6.69it/s]

    The radius of curvature is: 464.27510611744077 m 999.731848517913 m
    

     47%|█████████████████████████████████████                                          | 591/1261 [01:28<01:40,  6.69it/s]

    The radius of curvature is: 1099.56061707635 m 1128.44855004467 m
    

     47%|█████████████████████████████████████                                          | 592/1261 [01:28<01:39,  6.69it/s]

    The radius of curvature is: 557.8040891274579 m 9134.883019717074 m
    

     47%|█████████████████████████████████████▏                                         | 593/1261 [01:28<01:39,  6.69it/s]

    The radius of curvature is: 457.5240417433417 m 24219.465520944213 m
    

     47%|█████████████████████████████████████▏                                         | 594/1261 [01:28<01:39,  6.69it/s]

    The radius of curvature is: 14.836883419384117 m 6990.106936981604 m
    

     47%|█████████████████████████████████████▎                                         | 595/1261 [01:28<01:39,  6.69it/s]

    The radius of curvature is: 265.9724784624823 m 1386.0360364835067 m
    

     47%|█████████████████████████████████████▎                                         | 596/1261 [01:29<01:39,  6.69it/s]

    The radius of curvature is: 296.93118121034786 m 1208.9120801141532 m
    

     47%|█████████████████████████████████████▍                                         | 597/1261 [01:29<01:39,  6.69it/s]

    The radius of curvature is: 228.42114110375942 m 559.4177569162578 m
    

     47%|█████████████████████████████████████▍                                         | 598/1261 [01:29<01:39,  6.69it/s]

    The radius of curvature is: 100.2729760835528 m 1021.8271947916378 m
    

     48%|█████████████████████████████████████▌                                         | 599/1261 [01:29<01:38,  6.69it/s]

    The radius of curvature is: 7.420785402295684 m 839.1581505705958 m
    

     48%|█████████████████████████████████████▌                                         | 600/1261 [01:29<01:38,  6.69it/s]

    The radius of curvature is: 327.76255400271685 m 850.6622117027214 m
    

     48%|█████████████████████████████████████▋                                         | 601/1261 [01:29<01:38,  6.69it/s]

    The radius of curvature is: 55.150799340921836 m 837.4694676423744 m
    

     48%|█████████████████████████████████████▋                                         | 602/1261 [01:29<01:38,  6.69it/s]

    The radius of curvature is: 49.24382390473726 m 652.7249219754385 m
    

     48%|█████████████████████████████████████▊                                         | 603/1261 [01:30<01:38,  6.69it/s]

    The radius of curvature is: 53.95170552994556 m 72112.52427783371 m
    

     48%|█████████████████████████████████████▊                                         | 604/1261 [01:30<01:38,  6.69it/s]

    The radius of curvature is: 780.779127192709 m 4023.8941908884453 m
    

     48%|█████████████████████████████████████▉                                         | 605/1261 [01:30<01:37,  6.69it/s]

    The radius of curvature is: 807.5052664263566 m 2388.7046923366743 m
    

     48%|█████████████████████████████████████▉                                         | 606/1261 [01:30<01:37,  6.70it/s]

    The radius of curvature is: 692.2840117758267 m 81094.02721190936 m
    

     48%|██████████████████████████████████████                                         | 607/1261 [01:30<01:37,  6.70it/s]

    The radius of curvature is: 364.54789181165324 m 14378.118945761467 m
    

     48%|██████████████████████████████████████                                         | 608/1261 [01:30<01:37,  6.70it/s]

    The radius of curvature is: 221.30712934100774 m 4137.0250182216805 m
    

     48%|██████████████████████████████████████▏                                        | 609/1261 [01:30<01:37,  6.70it/s]

    The radius of curvature is: 184.49086409372413 m 1390.4734255566434 m
    

     48%|██████████████████████████████████████▏                                        | 610/1261 [01:31<01:37,  6.70it/s]

    The radius of curvature is: 213.69880913365824 m 924.516635449421 m
    

     48%|██████████████████████████████████████▎                                        | 611/1261 [01:31<01:37,  6.70it/s]

    The radius of curvature is: 454.6059201701755 m 671.0867349025295 m
    

     49%|██████████████████████████████████████▎                                        | 612/1261 [01:31<01:36,  6.70it/s]

    The radius of curvature is: 378.0535868618732 m 463.1835720937153 m
    

     49%|██████████████████████████████████████▍                                        | 613/1261 [01:31<01:36,  6.70it/s]

    The radius of curvature is: 488.5813985855011 m 429.22261529466107 m
    

     49%|██████████████████████████████████████▍                                        | 614/1261 [01:31<01:36,  6.70it/s]

    The radius of curvature is: 462.89258073992704 m 386.6577848413704 m
    

     49%|██████████████████████████████████████▌                                        | 615/1261 [01:31<01:36,  6.70it/s]

    The radius of curvature is: 549.1439057006667 m 586.3810460283779 m
    

     49%|██████████████████████████████████████▌                                        | 616/1261 [01:31<01:36,  6.70it/s]

    The radius of curvature is: 1068.082555318564 m 1253.1668818801172 m
    

     49%|██████████████████████████████████████▋                                        | 617/1261 [01:32<01:36,  6.70it/s]

    The radius of curvature is: 1452.533298189952 m 4639.994506969628 m
    

     49%|██████████████████████████████████████▋                                        | 618/1261 [01:32<01:36,  6.69it/s]

    The radius of curvature is: 1883.6875682817897 m 1042.122976018018 m
    

     49%|██████████████████████████████████████▊                                        | 619/1261 [01:32<01:35,  6.69it/s]

    The radius of curvature is: 4974.685641843878 m 1319.8420221899958 m
    

     49%|██████████████████████████████████████▊                                        | 620/1261 [01:32<01:35,  6.69it/s]

    The radius of curvature is: 2556.4132119666237 m 884.8173541971015 m
    

     49%|██████████████████████████████████████▉                                        | 621/1261 [01:32<01:35,  6.69it/s]

    The radius of curvature is: 2827.7873074239856 m 748.0695159364287 m
    

     49%|██████████████████████████████████████▉                                        | 622/1261 [01:32<01:35,  6.69it/s]

    The radius of curvature is: 978.8479463935641 m 786.3488950037749 m
    

     49%|███████████████████████████████████████                                        | 623/1261 [01:33<01:35,  6.69it/s]

    The radius of curvature is: 898.366583884078 m 1980.3125552306783 m
    

     49%|███████████████████████████████████████                                        | 624/1261 [01:33<01:35,  6.69it/s]

    The radius of curvature is: 781.2519485449452 m 1653.7621839759865 m
    

     50%|███████████████████████████████████████▏                                       | 625/1261 [01:33<01:35,  6.69it/s]

    The radius of curvature is: 803.3573318454742 m 1837.3529321302885 m
    

     50%|███████████████████████████████████████▏                                       | 626/1261 [01:33<01:34,  6.69it/s]

    The radius of curvature is: 870.97284218114 m 5899.103276880632 m
    

     50%|███████████████████████████████████████▎                                       | 627/1261 [01:33<01:34,  6.69it/s]

    The radius of curvature is: 868.4440987169181 m 2768.205738544893 m
    

     50%|███████████████████████████████████████▎                                       | 628/1261 [01:33<01:34,  6.69it/s]

    The radius of curvature is: 1011.9332195759732 m 1320.284943428911 m
    

     50%|███████████████████████████████████████▍                                       | 629/1261 [01:33<01:34,  6.69it/s]

    The radius of curvature is: 1207.4121324911423 m 68145.57021552688 m
    

     50%|███████████████████████████████████████▍                                       | 630/1261 [01:34<01:34,  6.69it/s]

    The radius of curvature is: 1593.4562218543583 m 6720.3196883808005 m
    

     50%|███████████████████████████████████████▌                                       | 631/1261 [01:34<01:34,  6.69it/s]

    The radius of curvature is: 1520.4063572425769 m 5626.955626609934 m
    

     50%|███████████████████████████████████████▌                                       | 632/1261 [01:34<01:33,  6.69it/s]

    The radius of curvature is: 1602.9177943140253 m 2127.954263711169 m
    

     50%|███████████████████████████████████████▋                                       | 633/1261 [01:34<01:33,  6.69it/s]

    The radius of curvature is: 1412.3899605196204 m 1014.1690825328933 m
    

     50%|███████████████████████████████████████▋                                       | 634/1261 [01:34<01:33,  6.69it/s]

    The radius of curvature is: 1218.0824349987686 m 650.7940975704709 m
    

     50%|███████████████████████████████████████▊                                       | 635/1261 [01:34<01:33,  6.69it/s]

    The radius of curvature is: 1122.244569780345 m 1053.0432745346739 m
    

     50%|███████████████████████████████████████▊                                       | 636/1261 [01:35<01:33,  6.69it/s]

    The radius of curvature is: 1182.0311788683591 m 925.4234951817177 m
    

     51%|███████████████████████████████████████▉                                       | 637/1261 [01:35<01:33,  6.69it/s]

    The radius of curvature is: 1293.067448336837 m 1329.9649691358243 m
    

     51%|███████████████████████████████████████▉                                       | 638/1261 [01:35<01:33,  6.69it/s]

    The radius of curvature is: 1470.2891869036762 m 1319.4832518614128 m
    

     51%|████████████████████████████████████████                                       | 639/1261 [01:35<01:32,  6.69it/s]

    The radius of curvature is: 1606.1238453459152 m 1189.0517875135326 m
    

     51%|████████████████████████████████████████                                       | 640/1261 [01:35<01:32,  6.69it/s]

    The radius of curvature is: 1610.534248137805 m 1130.0219747722545 m
    

     51%|████████████████████████████████████████▏                                      | 641/1261 [01:35<01:32,  6.69it/s]

    The radius of curvature is: 1643.7765727388155 m 1003.4443735410542 m
    

     51%|████████████████████████████████████████▏                                      | 642/1261 [01:35<01:32,  6.69it/s]

    The radius of curvature is: 1868.447009985446 m 1834.848825505558 m
    

     51%|████████████████████████████████████████▎                                      | 643/1261 [01:36<01:32,  6.69it/s]

    The radius of curvature is: 1943.6915185458681 m 1837.928025849801 m
    

     51%|████████████████████████████████████████▎                                      | 644/1261 [01:36<01:32,  6.69it/s]

    The radius of curvature is: 1859.8222942692546 m 1410.9034918449818 m
    

     51%|████████████████████████████████████████▍                                      | 645/1261 [01:36<01:32,  6.69it/s]

    The radius of curvature is: 2475.431743365589 m 932.0736606773706 m
    

     51%|████████████████████████████████████████▍                                      | 646/1261 [01:36<01:31,  6.69it/s]

    The radius of curvature is: 2135.120297210847 m 936.138055027946 m
    

     51%|████████████████████████████████████████▌                                      | 647/1261 [01:36<01:31,  6.69it/s]

    The radius of curvature is: 2349.481903453386 m 1154.7525059851728 m
    

     51%|████████████████████████████████████████▌                                      | 648/1261 [01:36<01:31,  6.69it/s]

    The radius of curvature is: 2277.6464869398524 m 1269.8822951569377 m
    

     51%|████████████████████████████████████████▋                                      | 649/1261 [01:36<01:31,  6.69it/s]

    The radius of curvature is: 2565.3650339251117 m 1001.281767403971 m
    

     52%|████████████████████████████████████████▋                                      | 650/1261 [01:37<01:31,  6.69it/s]

    The radius of curvature is: 2835.930336882927 m 973.5168875988805 m
    

     52%|████████████████████████████████████████▊                                      | 651/1261 [01:37<01:31,  6.69it/s]

    The radius of curvature is: 2855.8580197658393 m 917.4453835247671 m
    

     52%|████████████████████████████████████████▊                                      | 652/1261 [01:37<01:31,  6.69it/s]

    The radius of curvature is: 2516.969362942976 m 1293.0274325324724 m
    

     52%|████████████████████████████████████████▉                                      | 653/1261 [01:37<01:30,  6.69it/s]

    The radius of curvature is: 1938.6542654394816 m 2048.316256465175 m
    

     52%|████████████████████████████████████████▉                                      | 654/1261 [01:37<01:30,  6.69it/s]

    The radius of curvature is: 1898.6254010827947 m 2572.0137408013234 m
    

     52%|█████████████████████████████████████████                                      | 655/1261 [01:37<01:30,  6.69it/s]

    The radius of curvature is: 1459.2218798662982 m 1108.4128225989386 m
    

     52%|█████████████████████████████████████████                                      | 656/1261 [01:38<01:30,  6.69it/s]

    The radius of curvature is: 1317.6216560832286 m 1566.1103637066717 m
    

     52%|█████████████████████████████████████████▏                                     | 657/1261 [01:38<01:30,  6.69it/s]

    The radius of curvature is: 1264.7931831374312 m 996.5930198917175 m
    

     52%|█████████████████████████████████████████▏                                     | 658/1261 [01:38<01:30,  6.69it/s]

    The radius of curvature is: 1221.5429159963153 m 713.4113834350467 m
    

     52%|█████████████████████████████████████████▎                                     | 659/1261 [01:38<01:29,  6.69it/s]

    The radius of curvature is: 1094.6142156715837 m 782.9383863071479 m
    

     52%|█████████████████████████████████████████▎                                     | 660/1261 [01:38<01:29,  6.69it/s]

    The radius of curvature is: 1121.4305086278698 m 1624.2735152399775 m
    

     52%|█████████████████████████████████████████▍                                     | 661/1261 [01:38<01:29,  6.69it/s]

    The radius of curvature is: 1138.4803302130642 m 1944.7156778300914 m
    

     52%|█████████████████████████████████████████▍                                     | 662/1261 [01:38<01:29,  6.69it/s]

    The radius of curvature is: 1226.4188173777081 m 1712.880871787746 m
    

     53%|█████████████████████████████████████████▌                                     | 663/1261 [01:39<01:29,  6.69it/s]

    The radius of curvature is: 1166.0885617340011 m 1886.3099097993393 m
    

     53%|█████████████████████████████████████████▌                                     | 664/1261 [01:39<01:29,  6.69it/s]

    The radius of curvature is: 1180.7623382155355 m 1824.6375345716513 m
    

     53%|█████████████████████████████████████████▋                                     | 665/1261 [01:39<01:29,  6.69it/s]

    The radius of curvature is: 1090.7961380205063 m 4416.472389263419 m
    

     53%|█████████████████████████████████████████▋                                     | 666/1261 [01:39<01:28,  6.69it/s]

    The radius of curvature is: 1226.0312520515404 m 7182.797181690971 m
    

     53%|█████████████████████████████████████████▊                                     | 667/1261 [01:39<01:28,  6.69it/s]

    The radius of curvature is: 1263.3843488634282 m 9024.141642762783 m
    

     53%|█████████████████████████████████████████▊                                     | 668/1261 [01:39<01:28,  6.69it/s]

    The radius of curvature is: 1264.7838738857918 m 987.9975662287059 m
    

     53%|█████████████████████████████████████████▉                                     | 669/1261 [01:40<01:28,  6.69it/s]

    The radius of curvature is: 1363.657387092395 m 1403.5722656939304 m
    

     53%|█████████████████████████████████████████▉                                     | 670/1261 [01:40<01:28,  6.69it/s]

    The radius of curvature is: 1470.7086970588239 m 977.3433446352415 m
    

     53%|██████████████████████████████████████████                                     | 671/1261 [01:40<01:28,  6.69it/s]

    The radius of curvature is: 1551.277755504512 m 596.2861750201117 m
    

     53%|██████████████████████████████████████████                                     | 672/1261 [01:40<01:28,  6.69it/s]

    The radius of curvature is: 1700.0090060594596 m 883.0497693357847 m
    

     53%|██████████████████████████████████████████▏                                    | 673/1261 [01:40<01:27,  6.69it/s]

    The radius of curvature is: 1845.2185700412915 m 2397.1691624793884 m
    

     53%|██████████████████████████████████████████▏                                    | 674/1261 [01:40<01:27,  6.69it/s]

    The radius of curvature is: 1958.305403551886 m 1368.1258198464977 m
    

     54%|██████████████████████████████████████████▎                                    | 675/1261 [01:40<01:27,  6.69it/s]

    The radius of curvature is: 1977.6689110049822 m 1141.09695404959 m
    

     54%|██████████████████████████████████████████▎                                    | 676/1261 [01:41<01:27,  6.69it/s]

    The radius of curvature is: 2336.892906670378 m 1293.7112393478037 m
    

     54%|██████████████████████████████████████████▍                                    | 677/1261 [01:41<01:27,  6.69it/s]

    The radius of curvature is: 2039.3743617618622 m 1668.5585194804828 m
    

     54%|██████████████████████████████████████████▍                                    | 678/1261 [01:41<01:27,  6.69it/s]

    The radius of curvature is: 2205.9697973767366 m 2040.9008796029111 m
    

     54%|██████████████████████████████████████████▌                                    | 679/1261 [01:41<01:27,  6.69it/s]

    The radius of curvature is: 1966.4399330368026 m 1396.401041053006 m
    

     54%|██████████████████████████████████████████▌                                    | 680/1261 [01:41<01:26,  6.69it/s]

    The radius of curvature is: 1675.7158022005035 m 2067.632842072226 m
    

     54%|██████████████████████████████████████████▋                                    | 681/1261 [01:41<01:26,  6.69it/s]

    The radius of curvature is: 2140.6003732784416 m 2119.485465987861 m
    

     54%|██████████████████████████████████████████▋                                    | 682/1261 [01:41<01:26,  6.69it/s]

    The radius of curvature is: 1891.5908691425109 m 956.5710497026371 m
    

     54%|██████████████████████████████████████████▊                                    | 683/1261 [01:42<01:26,  6.69it/s]

    The radius of curvature is: 1806.2608310788762 m 943.9352564333026 m
    

     54%|██████████████████████████████████████████▊                                    | 684/1261 [01:42<01:26,  6.69it/s]

    The radius of curvature is: 2073.050029597363 m 685.7615743999306 m
    

     54%|██████████████████████████████████████████▉                                    | 685/1261 [01:42<01:26,  6.69it/s]

    The radius of curvature is: 2222.418375923298 m 1264.1745024026002 m
    

     54%|██████████████████████████████████████████▉                                    | 686/1261 [01:42<01:25,  6.69it/s]

    The radius of curvature is: 2371.5121382605316 m 1317.435715012186 m
    

     54%|███████████████████████████████████████████                                    | 687/1261 [01:42<01:25,  6.69it/s]

    The radius of curvature is: 2329.835853591976 m 907.5007589638378 m
    

     55%|███████████████████████████████████████████                                    | 688/1261 [01:42<01:25,  6.69it/s]

    The radius of curvature is: 2744.8089071201607 m 1370.0237124591563 m
    

     55%|███████████████████████████████████████████▏                                   | 689/1261 [01:43<01:25,  6.69it/s]

    The radius of curvature is: 3086.379615553225 m 1053.8378404361797 m
    

     55%|███████████████████████████████████████████▏                                   | 690/1261 [01:43<01:25,  6.69it/s]

    The radius of curvature is: 2103.0355873250805 m 1793.531169594852 m
    

     55%|███████████████████████████████████████████▎                                   | 691/1261 [01:43<01:25,  6.69it/s]

    The radius of curvature is: 1776.5178710916157 m 4872.147238446967 m
    

     55%|███████████████████████████████████████████▎                                   | 692/1261 [01:43<01:25,  6.69it/s]

    The radius of curvature is: 1457.3646839781964 m 1774.899843616739 m
    

     55%|███████████████████████████████████████████▍                                   | 693/1261 [01:43<01:24,  6.69it/s]

    The radius of curvature is: 1337.526809143871 m 7524.715983531957 m
    

     55%|███████████████████████████████████████████▍                                   | 694/1261 [01:43<01:24,  6.69it/s]

    The radius of curvature is: 1276.7823653101814 m 2048.656893468605 m
    

     55%|███████████████████████████████████████████▌                                   | 695/1261 [01:43<01:24,  6.69it/s]

    The radius of curvature is: 1155.4329438818586 m 770.2401854939839 m
    

     55%|███████████████████████████████████████████▌                                   | 696/1261 [01:44<01:24,  6.69it/s]

    The radius of curvature is: 1113.1665888358548 m 2165.40115206364 m
    

     55%|███████████████████████████████████████████▋                                   | 697/1261 [01:44<01:24,  6.69it/s]

    The radius of curvature is: 1048.9660473220517 m 1658.051288759125 m
    

     55%|███████████████████████████████████████████▋                                   | 698/1261 [01:44<01:24,  6.69it/s]

    The radius of curvature is: 1031.3192167380757 m 2408.106536447801 m
    

     55%|███████████████████████████████████████████▊                                   | 699/1261 [01:44<01:24,  6.69it/s]

    The radius of curvature is: 1117.1442056653484 m 1452.1118661763503 m
    

     56%|███████████████████████████████████████████▊                                   | 700/1261 [01:44<01:23,  6.69it/s]

    The radius of curvature is: 1158.830220516303 m 1495.2971183361594 m
    

     56%|███████████████████████████████████████████▉                                   | 701/1261 [01:44<01:23,  6.69it/s]

    The radius of curvature is: 1256.440057316219 m 1485.428208513781 m
    

     56%|███████████████████████████████████████████▉                                   | 702/1261 [01:44<01:23,  6.69it/s]

    The radius of curvature is: 1260.646377108006 m 1265.7276875354933 m
    

     56%|████████████████████████████████████████████                                   | 703/1261 [01:45<01:23,  6.69it/s]

    The radius of curvature is: 1247.8192278984468 m 1857.6512036769147 m
    

     56%|████████████████████████████████████████████                                   | 704/1261 [01:45<01:23,  6.69it/s]

    The radius of curvature is: 1368.6335436899847 m 1261.95847098281 m
    

     56%|████████████████████████████████████████████▏                                  | 705/1261 [01:45<01:23,  6.69it/s]

    The radius of curvature is: 1507.7098401499338 m 832.0358838547796 m
    

     56%|████████████████████████████████████████████▏                                  | 706/1261 [01:45<01:22,  6.69it/s]

    The radius of curvature is: 1406.500646252298 m 1070.6793252982266 m
    

     56%|████████████████████████████████████████████▎                                  | 707/1261 [01:45<01:22,  6.69it/s]

    The radius of curvature is: 1360.65624984855 m 842.1965231825287 m
    

     56%|████████████████████████████████████████████▎                                  | 708/1261 [01:45<01:22,  6.69it/s]

    The radius of curvature is: 1438.4486033317319 m 995.1552669049095 m
    

     56%|████████████████████████████████████████████▍                                  | 709/1261 [01:46<01:22,  6.69it/s]

    The radius of curvature is: 1299.2688467028695 m 1269.3232576892062 m
    

     56%|████████████████████████████████████████████▍                                  | 710/1261 [01:46<01:22,  6.69it/s]

    The radius of curvature is: 1396.972312681233 m 1366.7399090531148 m
    

     56%|████████████████████████████████████████████▌                                  | 711/1261 [01:46<01:22,  6.69it/s]

    The radius of curvature is: 1417.4634333837014 m 876.4615106709907 m
    

     56%|████████████████████████████████████████████▌                                  | 712/1261 [01:46<01:22,  6.69it/s]

    The radius of curvature is: 1503.529074492185 m 1054.6694720315147 m
    

     57%|████████████████████████████████████████████▋                                  | 713/1261 [01:46<01:21,  6.69it/s]

    The radius of curvature is: 1621.4563760254875 m 1532.5727884396633 m
    

     57%|████████████████████████████████████████████▋                                  | 714/1261 [01:46<01:21,  6.69it/s]

    The radius of curvature is: 1398.7186341901786 m 1346.0606177152013 m
    

     57%|████████████████████████████████████████████▊                                  | 715/1261 [01:46<01:21,  6.69it/s]

    The radius of curvature is: 1415.122854773593 m 1156.1997530713547 m
    

     57%|████████████████████████████████████████████▊                                  | 716/1261 [01:47<01:21,  6.69it/s]

    The radius of curvature is: 1574.9397843751362 m 1649.0129544696701 m
    

     57%|████████████████████████████████████████████▉                                  | 717/1261 [01:47<01:21,  6.69it/s]

    The radius of curvature is: 1604.167920646176 m 1634.850297317636 m
    

     57%|████████████████████████████████████████████▉                                  | 718/1261 [01:47<01:21,  6.69it/s]

    The radius of curvature is: 1936.4766214057872 m 1438.8356475588287 m
    

     57%|█████████████████████████████████████████████                                  | 719/1261 [01:47<01:21,  6.69it/s]

    The radius of curvature is: 1924.090199865818 m 846.3056384890921 m
    

     57%|█████████████████████████████████████████████                                  | 720/1261 [01:47<01:20,  6.69it/s]

    The radius of curvature is: 2103.8026363678155 m 560.0635237313373 m
    

     57%|█████████████████████████████████████████████▏                                 | 721/1261 [01:47<01:20,  6.69it/s]

    The radius of curvature is: 2314.065753897264 m 658.2016693296425 m
    

     57%|█████████████████████████████████████████████▏                                 | 722/1261 [01:47<01:20,  6.69it/s]

    The radius of curvature is: 2224.829906966106 m 615.8744922382458 m
    

     57%|█████████████████████████████████████████████▎                                 | 723/1261 [01:48<01:20,  6.69it/s]

    The radius of curvature is: 2241.3490604018552 m 1618.3140145329132 m
    

     57%|█████████████████████████████████████████████▎                                 | 724/1261 [01:48<01:20,  6.69it/s]

    The radius of curvature is: 2118.5120890040316 m 1298.903884281189 m
    

     57%|█████████████████████████████████████████████▍                                 | 725/1261 [01:48<01:20,  6.69it/s]

    The radius of curvature is: 1959.4972874255498 m 1259.419372838726 m
    

     58%|█████████████████████████████████████████████▍                                 | 726/1261 [01:48<01:19,  6.69it/s]

    The radius of curvature is: 1685.5577495543512 m 999.3640923651245 m
    

     58%|█████████████████████████████████████████████▌                                 | 727/1261 [01:48<01:19,  6.69it/s]

    The radius of curvature is: 1505.0998068034824 m 1398.7076035715302 m
    

     58%|█████████████████████████████████████████████▌                                 | 728/1261 [01:48<01:19,  6.69it/s]

    The radius of curvature is: 1231.9245100709898 m 1894.9275593869645 m
    

     58%|█████████████████████████████████████████████▋                                 | 729/1261 [01:48<01:19,  6.69it/s]

    The radius of curvature is: 1278.6417246759008 m 1579.3893152345397 m
    

     58%|█████████████████████████████████████████████▋                                 | 730/1261 [01:49<01:19,  6.69it/s]

    The radius of curvature is: 1160.0574635630815 m 1851.7672783535606 m
    

     58%|█████████████████████████████████████████████▊                                 | 731/1261 [01:49<01:19,  6.69it/s]

    The radius of curvature is: 1196.9343392538517 m 963.0302619742674 m
    

     58%|█████████████████████████████████████████████▊                                 | 732/1261 [01:49<01:19,  6.69it/s]

    The radius of curvature is: 1120.6834495810574 m 795.1227795110071 m
    

     58%|█████████████████████████████████████████████▉                                 | 733/1261 [01:49<01:18,  6.69it/s]

    The radius of curvature is: 1127.868128946032 m 899.1278562767823 m
    

     58%|█████████████████████████████████████████████▉                                 | 734/1261 [01:49<01:18,  6.69it/s]

    The radius of curvature is: 1174.5420051065837 m 2132.6413259421197 m
    

     58%|██████████████████████████████████████████████                                 | 735/1261 [01:49<01:18,  6.69it/s]

    The radius of curvature is: 1248.8361432672757 m 1752.2501935856853 m
    

     58%|██████████████████████████████████████████████                                 | 736/1261 [01:50<01:18,  6.69it/s]

    The radius of curvature is: 1291.82185749079 m 1074.0111747268672 m
    

     58%|██████████████████████████████████████████████▏                                | 737/1261 [01:50<01:18,  6.69it/s]

    The radius of curvature is: 1416.5825128666102 m 1876.4022803891298 m
    

     59%|██████████████████████████████████████████████▏                                | 738/1261 [01:50<01:18,  6.69it/s]

    The radius of curvature is: 1399.7569741912882 m 2371.235784100849 m
    

     59%|██████████████████████████████████████████████▎                                | 739/1261 [01:50<01:18,  6.69it/s]

    The radius of curvature is: 1250.7644966872808 m 2562.1460770703006 m
    

     59%|██████████████████████████████████████████████▎                                | 740/1261 [01:50<01:17,  6.69it/s]

    The radius of curvature is: 1258.999356716895 m 1558.0375064841062 m
    

     59%|██████████████████████████████████████████████▍                                | 741/1261 [01:50<01:17,  6.69it/s]

    The radius of curvature is: 1201.8334456386926 m 1132.0603147210754 m
    

     59%|██████████████████████████████████████████████▍                                | 742/1261 [01:50<01:17,  6.69it/s]

    The radius of curvature is: 1283.3912491068056 m 958.371622668974 m
    

     59%|██████████████████████████████████████████████▌                                | 743/1261 [01:51<01:17,  6.69it/s]

    The radius of curvature is: 1371.6276144026365 m 908.3202074411946 m
    

     59%|██████████████████████████████████████████████▌                                | 744/1261 [01:51<01:17,  6.69it/s]

    The radius of curvature is: 1462.356129787998 m 673.053481927032 m
    

     59%|██████████████████████████████████████████████▋                                | 745/1261 [01:51<01:17,  6.69it/s]

    The radius of curvature is: 1428.2007150818342 m 1104.848469478899 m
    

     59%|██████████████████████████████████████████████▋                                | 746/1261 [01:51<01:16,  6.69it/s]

    The radius of curvature is: 1566.0314019681198 m 1507.6084022979114 m
    

     59%|██████████████████████████████████████████████▊                                | 747/1261 [01:51<01:16,  6.69it/s]

    The radius of curvature is: 1811.777209638088 m 1207.9230246068687 m
    

     59%|██████████████████████████████████████████████▊                                | 748/1261 [01:51<01:16,  6.69it/s]

    The radius of curvature is: 2077.6907042768285 m 859.7893722486128 m
    

     59%|██████████████████████████████████████████████▉                                | 749/1261 [01:51<01:16,  6.69it/s]

    The radius of curvature is: 1863.9003466343745 m 977.2486791611194 m
    

     59%|██████████████████████████████████████████████▉                                | 750/1261 [01:52<01:16,  6.69it/s]

    The radius of curvature is: 2292.37283652268 m 1107.0157619797378 m
    

     60%|███████████████████████████████████████████████                                | 751/1261 [01:52<01:16,  6.69it/s]

    The radius of curvature is: 1787.6070197291383 m 1634.9097987028929 m
    

     60%|███████████████████████████████████████████████                                | 752/1261 [01:52<01:16,  6.69it/s]

    The radius of curvature is: 1683.403489973137 m 2077.564193407051 m
    

     60%|███████████████████████████████████████████████▏                               | 753/1261 [01:52<01:15,  6.69it/s]

    The radius of curvature is: 1704.7551606587958 m 1258.4205969487346 m
    

     60%|███████████████████████████████████████████████▏                               | 754/1261 [01:52<01:15,  6.69it/s]

    The radius of curvature is: 1810.3301681790604 m 980.9817793836929 m
    

     60%|███████████████████████████████████████████████▎                               | 755/1261 [01:52<01:15,  6.69it/s]

    The radius of curvature is: 1775.2486443123255 m 627.4803927928651 m
    

     60%|███████████████████████████████████████████████▎                               | 756/1261 [01:53<01:15,  6.69it/s]

    The radius of curvature is: 1881.515358810658 m 1224.7045515591885 m
    

     60%|███████████████████████████████████████████████▍                               | 757/1261 [01:53<01:15,  6.69it/s]

    The radius of curvature is: 1756.813354293434 m 778.4211295591164 m
    

     60%|███████████████████████████████████████████████▍                               | 758/1261 [01:53<01:15,  6.69it/s]

    The radius of curvature is: 1965.6560002489973 m 1137.8298829436678 m
    

     60%|███████████████████████████████████████████████▌                               | 759/1261 [01:53<01:15,  6.69it/s]

    The radius of curvature is: 2003.2432882736869 m 1314.3400463342425 m
    

     60%|███████████████████████████████████████████████▌                               | 760/1261 [01:53<01:14,  6.69it/s]

    The radius of curvature is: 2447.39956166106 m 1052.354473127854 m
    

     60%|███████████████████████████████████████████████▋                               | 761/1261 [01:53<01:14,  6.69it/s]

    The radius of curvature is: 2641.638081830214 m 1280.869632029491 m
    

     60%|███████████████████████████████████████████████▋                               | 762/1261 [01:53<01:14,  6.69it/s]

    The radius of curvature is: 2213.0792608429833 m 864.722878933715 m
    

     61%|███████████████████████████████████████████████▊                               | 763/1261 [01:54<01:14,  6.69it/s]

    The radius of curvature is: 1677.128641350648 m 1122.3698971385056 m
    

     61%|███████████████████████████████████████████████▊                               | 764/1261 [01:54<01:14,  6.68it/s]

    The radius of curvature is: 1593.5297083990047 m 3503.9016357772994 m
    

     61%|███████████████████████████████████████████████▉                               | 765/1261 [01:54<01:14,  6.68it/s]

    The radius of curvature is: 1300.8955858000443 m 2906.2629559861984 m
    

     61%|███████████████████████████████████████████████▉                               | 766/1261 [01:54<01:14,  6.68it/s]

    The radius of curvature is: 1215.7936924155986 m 1768.316168637359 m
    

     61%|████████████████████████████████████████████████                               | 767/1261 [01:54<01:13,  6.68it/s]

    The radius of curvature is: 1128.2451395958853 m 1760.2632068435266 m
    

     61%|████████████████████████████████████████████████                               | 768/1261 [01:54<01:13,  6.68it/s]

    The radius of curvature is: 1092.3978315009017 m 1126.0673610073154 m
    

     61%|████████████████████████████████████████████████▏                              | 769/1261 [01:55<01:13,  6.68it/s]

    The radius of curvature is: 990.384396531319 m 825.2804118177246 m
    

     61%|████████████████████████████████████████████████▏                              | 770/1261 [01:55<01:13,  6.68it/s]

    The radius of curvature is: 1045.547882345103 m 2930.94457500221 m
    

     61%|████████████████████████████████████████████████▎                              | 771/1261 [01:55<01:13,  6.68it/s]

    The radius of curvature is: 1107.7348387186544 m 1820.9385902605773 m
    

     61%|████████████████████████████████████████████████▎                              | 772/1261 [01:55<01:13,  6.68it/s]

    The radius of curvature is: 1154.1783555750494 m 1420.5206354561515 m
    

     61%|████████████████████████████████████████████████▍                              | 773/1261 [01:55<01:13,  6.68it/s]

    The radius of curvature is: 1226.576378766014 m 1196.492292621871 m
    

     61%|████████████████████████████████████████████████▍                              | 774/1261 [01:55<01:12,  6.68it/s]

    The radius of curvature is: 1153.6182308355194 m 1199.5970381048035 m
    

     61%|████████████████████████████████████████████████▌                              | 775/1261 [01:55<01:12,  6.68it/s]

    The radius of curvature is: 1173.4592014858756 m 1918.7730991645785 m
    

     62%|████████████████████████████████████████████████▌                              | 776/1261 [01:56<01:12,  6.68it/s]

    The radius of curvature is: 1252.266102491658 m 2530.8606301300883 m
    

     62%|████████████████████████████████████████████████▋                              | 777/1261 [01:56<01:12,  6.68it/s]

    The radius of curvature is: 1323.6672584874577 m 1758.3874800998044 m
    

     62%|████████████████████████████████████████████████▋                              | 778/1261 [01:56<01:12,  6.68it/s]

    The radius of curvature is: 1417.3196050220592 m 1208.1052496758034 m
    

     62%|████████████████████████████████████████████████▊                              | 779/1261 [01:56<01:12,  6.68it/s]

    The radius of curvature is: 1411.6553490505012 m 942.9594185210568 m
    

     62%|████████████████████████████████████████████████▊                              | 780/1261 [01:56<01:11,  6.68it/s]

    The radius of curvature is: 1472.9933254860644 m 934.6680207950068 m
    

     62%|████████████████████████████████████████████████▉                              | 781/1261 [01:56<01:11,  6.68it/s]

    The radius of curvature is: 1650.4957894071931 m 750.5286610959862 m
    

     62%|████████████████████████████████████████████████▉                              | 782/1261 [01:57<01:11,  6.68it/s]

    The radius of curvature is: 1501.531873560125 m 1361.2234847811708 m
    

     62%|█████████████████████████████████████████████████                              | 783/1261 [01:57<01:11,  6.68it/s]

    The radius of curvature is: 1775.1005224768812 m 1153.1582276306976 m
    

     62%|█████████████████████████████████████████████████                              | 784/1261 [01:57<01:11,  6.68it/s]

    The radius of curvature is: 2137.014748714596 m 1112.8563513996276 m
    

     62%|█████████████████████████████████████████████████▏                             | 785/1261 [01:57<01:11,  6.68it/s]

    The radius of curvature is: 2287.3268561094865 m 821.587238176683 m
    

     62%|█████████████████████████████████████████████████▏                             | 786/1261 [01:57<01:11,  6.68it/s]

    The radius of curvature is: 2284.114348510345 m 1305.0560581624486 m
    

     62%|█████████████████████████████████████████████████▎                             | 787/1261 [01:57<01:10,  6.68it/s]

    The radius of curvature is: 1985.790640977014 m 4652.200326812821 m
    

     62%|█████████████████████████████████████████████████▎                             | 788/1261 [01:57<01:10,  6.68it/s]

    The radius of curvature is: 1652.317584877691 m 2699.7856985947024 m
    

     63%|█████████████████████████████████████████████████▍                             | 789/1261 [01:58<01:10,  6.68it/s]

    The radius of curvature is: 1391.2100240887685 m 1976.0470343650088 m
    

     63%|█████████████████████████████████████████████████▍                             | 790/1261 [01:58<01:10,  6.68it/s]

    The radius of curvature is: 1552.3040908872713 m 1708.950824103238 m
    

     63%|█████████████████████████████████████████████████▌                             | 791/1261 [01:58<01:10,  6.68it/s]

    The radius of curvature is: 1430.2975696413582 m 1002.7990258888309 m
    

     63%|█████████████████████████████████████████████████▌                             | 792/1261 [01:58<01:10,  6.68it/s]

    The radius of curvature is: 1437.3505118458793 m 814.1217641324075 m
    

     63%|█████████████████████████████████████████████████▋                             | 793/1261 [01:58<01:10,  6.68it/s]

    The radius of curvature is: 1600.0828610592175 m 767.853725026785 m
    

     63%|█████████████████████████████████████████████████▋                             | 794/1261 [01:58<01:09,  6.68it/s]

    The radius of curvature is: 1780.9612599429547 m 2157.975274114739 m
    

     63%|█████████████████████████████████████████████████▊                             | 795/1261 [01:58<01:09,  6.68it/s]

    The radius of curvature is: 2548.8869046011614 m 1163.5786982762086 m
    

     63%|█████████████████████████████████████████████████▊                             | 796/1261 [01:59<01:09,  6.68it/s]

    The radius of curvature is: 2934.710716887867 m 810.0251609205135 m
    

     63%|█████████████████████████████████████████████████▉                             | 797/1261 [01:59<01:09,  6.68it/s]

    The radius of curvature is: 4046.546113255459 m 2165.920073238999 m
    

     63%|█████████████████████████████████████████████████▉                             | 798/1261 [01:59<01:09,  6.68it/s]

    The radius of curvature is: 4210.191737842375 m 2374.596258737624 m
    

     63%|██████████████████████████████████████████████████                             | 799/1261 [01:59<01:09,  6.68it/s]

    The radius of curvature is: 3919.805264778824 m 2236.907755894831 m
    

     63%|██████████████████████████████████████████████████                             | 800/1261 [01:59<01:08,  6.68it/s]

    The radius of curvature is: 4722.182832767232 m 14430.305726905408 m
    

     64%|██████████████████████████████████████████████████▏                            | 801/1261 [01:59<01:08,  6.68it/s]

    The radius of curvature is: 3289.3910131136745 m 2729.708755754507 m
    

     64%|██████████████████████████████████████████████████▏                            | 802/1261 [02:00<01:08,  6.68it/s]

    The radius of curvature is: 2723.3397155725006 m 1598.8234038001797 m
    

     64%|██████████████████████████████████████████████████▎                            | 803/1261 [02:00<01:08,  6.68it/s]

    The radius of curvature is: 4669.524896113856 m 1576.1086254514667 m
    

     64%|██████████████████████████████████████████████████▎                            | 804/1261 [02:00<01:08,  6.68it/s]

    The radius of curvature is: 2877.8675056738234 m 1323.4304526493554 m
    

     64%|██████████████████████████████████████████████████▍                            | 805/1261 [02:00<01:08,  6.68it/s]

    The radius of curvature is: 2735.651328791311 m 4491.748216840746 m
    

     64%|██████████████████████████████████████████████████▍                            | 806/1261 [02:00<01:08,  6.68it/s]

    The radius of curvature is: 2603.256082794202 m 10715.626861727356 m
    

     64%|██████████████████████████████████████████████████▌                            | 807/1261 [02:00<01:07,  6.68it/s]

    The radius of curvature is: 2357.325828313623 m 2527.766986025665 m
    

     64%|██████████████████████████████████████████████████▌                            | 808/1261 [02:00<01:07,  6.68it/s]

    The radius of curvature is: 3100.8525444148436 m 3242.6067744773873 m
    

     64%|██████████████████████████████████████████████████▋                            | 809/1261 [02:01<01:07,  6.68it/s]

    The radius of curvature is: 3360.261121855909 m 2492.259169707142 m
    

     64%|██████████████████████████████████████████████████▋                            | 810/1261 [02:01<01:07,  6.68it/s]

    The radius of curvature is: 1937.7215774527351 m 9886.547941358116 m
    

     64%|██████████████████████████████████████████████████▊                            | 811/1261 [02:01<01:07,  6.68it/s]

    The radius of curvature is: 1671.9166452457464 m 10420.071435080588 m
    

     64%|██████████████████████████████████████████████████▊                            | 812/1261 [02:01<01:07,  6.68it/s]

    The radius of curvature is: 1550.5547621142628 m 6766.332810657946 m
    

     64%|██████████████████████████████████████████████████▉                            | 813/1261 [02:01<01:07,  6.68it/s]

    The radius of curvature is: 1315.1699960304009 m 3650.860816908408 m
    

     65%|██████████████████████████████████████████████████▉                            | 814/1261 [02:01<01:06,  6.68it/s]

    The radius of curvature is: 1224.3101801704802 m 164041.3394182748 m
    

     65%|███████████████████████████████████████████████████                            | 815/1261 [02:01<01:06,  6.68it/s]

    The radius of curvature is: 1252.4115582666223 m 590.3482259536197 m
    

     65%|███████████████████████████████████████████████████                            | 816/1261 [02:02<01:06,  6.68it/s]

    The radius of curvature is: 1205.9737935827868 m 586.6873245573393 m
    

     65%|███████████████████████████████████████████████████▏                           | 817/1261 [02:02<01:06,  6.68it/s]

    The radius of curvature is: 1219.0050749950942 m 1517.804827462315 m
    

     65%|███████████████████████████████████████████████████▏                           | 818/1261 [02:02<01:06,  6.68it/s]

    The radius of curvature is: 1349.979774864149 m 832.7277126405297 m
    

     65%|███████████████████████████████████████████████████▎                           | 819/1261 [02:02<01:06,  6.68it/s]

    The radius of curvature is: 1434.7746244912519 m 1050.4837528852606 m
    

     65%|███████████████████████████████████████████████████▎                           | 820/1261 [02:02<01:05,  6.68it/s]

    The radius of curvature is: 1702.0383541653935 m 1020.3403468464255 m
    

     65%|███████████████████████████████████████████████████▍                           | 821/1261 [02:02<01:05,  6.68it/s]

    The radius of curvature is: 1887.4544891215505 m 1170.1757684622203 m
    

     65%|███████████████████████████████████████████████████▍                           | 822/1261 [02:03<01:05,  6.68it/s]

    The radius of curvature is: 1986.8796856374372 m 873.5882837204839 m
    

     65%|███████████████████████████████████████████████████▌                           | 823/1261 [02:03<01:05,  6.68it/s]

    The radius of curvature is: 2301.31246490854 m 1583.266950410212 m
    

     65%|███████████████████████████████████████████████████▌                           | 824/1261 [02:03<01:05,  6.68it/s]

    The radius of curvature is: 2487.8484228879925 m 1344.5722490708986 m
    

     65%|███████████████████████████████████████████████████▋                           | 825/1261 [02:03<01:05,  6.68it/s]

    The radius of curvature is: 2646.498147149646 m 1601.273214677624 m
    

     66%|███████████████████████████████████████████████████▋                           | 826/1261 [02:03<01:05,  6.68it/s]

    The radius of curvature is: 2648.7002158944856 m 1532.0656801080866 m
    

     66%|███████████████████████████████████████████████████▊                           | 827/1261 [02:03<01:04,  6.68it/s]

    The radius of curvature is: 2607.147739111491 m 973.773415014869 m
    

     66%|███████████████████████████████████████████████████▊                           | 828/1261 [02:03<01:04,  6.68it/s]

    The radius of curvature is: 2872.1657730764896 m 1081.7383273479186 m
    

     66%|███████████████████████████████████████████████████▉                           | 829/1261 [02:04<01:04,  6.68it/s]

    The radius of curvature is: 3087.6314797382784 m 672.8570781388877 m
    

     66%|███████████████████████████████████████████████████▉                           | 830/1261 [02:04<01:04,  6.68it/s]

    The radius of curvature is: 2587.125326378589 m 1100.1580009347226 m
    

     66%|████████████████████████████████████████████████████                           | 831/1261 [02:04<01:04,  6.68it/s]

    The radius of curvature is: 2332.4816842318455 m 872.3227306743513 m
    

     66%|████████████████████████████████████████████████████                           | 832/1261 [02:04<01:04,  6.68it/s]

    The radius of curvature is: 2450.0334697466305 m 1007.9252369933803 m
    

     66%|████████████████████████████████████████████████████▏                          | 833/1261 [02:04<01:04,  6.68it/s]

    The radius of curvature is: 2368.0686732432296 m 1471.8549398283114 m
    

     66%|████████████████████████████████████████████████████▏                          | 834/1261 [02:04<01:03,  6.68it/s]

    The radius of curvature is: 1887.7362776530772 m 1059.0249055380764 m
    

     66%|████████████████████████████████████████████████████▎                          | 835/1261 [02:04<01:03,  6.68it/s]

    The radius of curvature is: 1894.8833715509707 m 1845.4140029675384 m
    

     66%|████████████████████████████████████████████████████▎                          | 836/1261 [02:05<01:03,  6.68it/s]

    The radius of curvature is: 1805.8623791653797 m 1505.1436879761047 m
    

     66%|████████████████████████████████████████████████████▍                          | 837/1261 [02:05<01:03,  6.68it/s]

    The radius of curvature is: 1650.7727234053348 m 1494.9416576073813 m
    

     66%|████████████████████████████████████████████████████▍                          | 838/1261 [02:05<01:03,  6.68it/s]

    The radius of curvature is: 1668.5703525384322 m 1800.4453033929237 m
    

     67%|████████████████████████████████████████████████████▌                          | 839/1261 [02:05<01:03,  6.68it/s]

    The radius of curvature is: 1372.9294169729292 m 1125.3056792883895 m
    

     67%|████████████████████████████████████████████████████▌                          | 840/1261 [02:05<01:03,  6.68it/s]

    The radius of curvature is: 1382.1192902435182 m 625.5187222466827 m
    

     67%|████████████████████████████████████████████████████▋                          | 841/1261 [02:05<01:02,  6.68it/s]

    The radius of curvature is: 1389.6774698847382 m 813.9662196040931 m
    

     67%|████████████████████████████████████████████████████▊                          | 842/1261 [02:06<01:02,  6.68it/s]

    The radius of curvature is: 1254.1074426366415 m 810.5462290346612 m
    

     67%|████████████████████████████████████████████████████▊                          | 843/1261 [02:06<01:02,  6.68it/s]

    The radius of curvature is: 1225.2489056818058 m 1522.267512079001 m
    

     67%|████████████████████████████████████████████████████▉                          | 844/1261 [02:06<01:02,  6.68it/s]

    The radius of curvature is: 1235.6031703561696 m 1455.2993366568144 m
    

     67%|████████████████████████████████████████████████████▉                          | 845/1261 [02:06<01:02,  6.68it/s]

    The radius of curvature is: 1246.5873919785147 m 1241.680744139733 m
    

     67%|█████████████████████████████████████████████████████                          | 846/1261 [02:06<01:02,  6.68it/s]

    The radius of curvature is: 1073.2466254318451 m 2731.702155206734 m
    

     67%|█████████████████████████████████████████████████████                          | 847/1261 [02:06<01:01,  6.68it/s]

    The radius of curvature is: 1091.1428270996826 m 1345.7215308273092 m
    

     67%|█████████████████████████████████████████████████████▏                         | 848/1261 [02:06<01:01,  6.68it/s]

    The radius of curvature is: 1150.1759341661993 m 6665.6853555213465 m
    

     67%|█████████████████████████████████████████████████████▏                         | 849/1261 [02:07<01:01,  6.68it/s]

    The radius of curvature is: 1128.4279528098175 m 5789.268729607255 m
    

     67%|█████████████████████████████████████████████████████▎                         | 850/1261 [02:07<01:01,  6.68it/s]

    The radius of curvature is: 1153.7118116982099 m 2265.7745032811804 m
    

     67%|█████████████████████████████████████████████████████▎                         | 851/1261 [02:07<01:01,  6.68it/s]

    The radius of curvature is: 1225.546925850288 m 2110.3160186811438 m
    

     68%|█████████████████████████████████████████████████████▍                         | 852/1261 [02:07<01:01,  6.68it/s]

    The radius of curvature is: 1288.9704555955516 m 824.3295347140142 m
    

     68%|█████████████████████████████████████████████████████▍                         | 853/1261 [02:07<01:01,  6.68it/s]

    The radius of curvature is: 1579.7814767564685 m 1685.9686669776768 m
    

     68%|█████████████████████████████████████████████████████▌                         | 854/1261 [02:07<01:00,  6.68it/s]

    The radius of curvature is: 2051.026929159622 m 1571.7124883720385 m
    

     68%|█████████████████████████████████████████████████████▌                         | 855/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 2431.907015457065 m 2240.6125478234953 m
    

     68%|█████████████████████████████████████████████████████▋                         | 856/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 3651.6866578689273 m 2142.7251437757677 m
    

     68%|█████████████████████████████████████████████████████▋                         | 857/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 2789.6871481099797 m 2061.9948069207794 m
    

     68%|█████████████████████████████████████████████████████▊                         | 858/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 3188.4130740435007 m 635.8294066207231 m
    

     68%|█████████████████████████████████████████████████████▊                         | 859/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 2802.592543125576 m 8672.27951977567 m
    

     68%|█████████████████████████████████████████████████████▉                         | 860/1261 [02:08<01:00,  6.68it/s]

    The radius of curvature is: 1983.9239790842128 m 2478.2628271308395 m
    

     68%|█████████████████████████████████████████████████████▉                         | 861/1261 [02:08<00:59,  6.68it/s]

    The radius of curvature is: 1798.4992741251017 m 1677.2034465372676 m
    

     68%|██████████████████████████████████████████████████████                         | 862/1261 [02:09<00:59,  6.68it/s]

    The radius of curvature is: 1724.119342245421 m 1497.7262445083093 m
    

     68%|██████████████████████████████████████████████████████                         | 863/1261 [02:09<00:59,  6.68it/s]

    The radius of curvature is: 1530.3083868233348 m 1233.0397852899941 m
    

     69%|██████████████████████████████████████████████████████▏                        | 864/1261 [02:09<00:59,  6.68it/s]

    The radius of curvature is: 1435.5069333138688 m 1051.7207685026303 m
    

     69%|██████████████████████████████████████████████████████▏                        | 865/1261 [02:09<00:59,  6.68it/s]

    The radius of curvature is: 1777.4382399466458 m 1312.183242409894 m
    

     69%|██████████████████████████████████████████████████████▎                        | 866/1261 [02:09<00:59,  6.68it/s]

    The radius of curvature is: 1835.6766283311968 m 679.036502767527 m
    

     69%|██████████████████████████████████████████████████████▎                        | 867/1261 [02:09<00:58,  6.68it/s]

    The radius of curvature is: 2133.6118460132193 m 1256.2386735111436 m
    

     69%|██████████████████████████████████████████████████████▍                        | 868/1261 [02:09<00:58,  6.68it/s]

    The radius of curvature is: 3190.410175662038 m 1146.330052537681 m
    

     69%|██████████████████████████████████████████████████████▍                        | 869/1261 [02:10<00:58,  6.68it/s]

    The radius of curvature is: 2801.8589820899856 m 1149.6815257711237 m
    

     69%|██████████████████████████████████████████████████████▌                        | 870/1261 [02:10<00:58,  6.68it/s]

    The radius of curvature is: 2653.5470304746173 m 1250.853165665058 m
    

     69%|██████████████████████████████████████████████████████▌                        | 871/1261 [02:10<00:58,  6.68it/s]

    The radius of curvature is: 2801.2729723536395 m 2110.843421006691 m
    

     69%|██████████████████████████████████████████████████████▋                        | 872/1261 [02:10<00:58,  6.68it/s]

    The radius of curvature is: 2190.2522368027403 m 2581.04098446502 m
    

     69%|██████████████████████████████████████████████████████▋                        | 873/1261 [02:10<00:58,  6.68it/s]

    The radius of curvature is: 1869.4212448994695 m 3230.566931452745 m
    

     69%|██████████████████████████████████████████████████████▊                        | 874/1261 [02:10<00:57,  6.68it/s]

    The radius of curvature is: 1997.0496788139594 m 2619.512421086729 m
    

     69%|██████████████████████████████████████████████████████▊                        | 875/1261 [02:11<00:57,  6.68it/s]

    The radius of curvature is: 1558.4962469145803 m 1056.5056181183097 m
    

     69%|██████████████████████████████████████████████████████▉                        | 876/1261 [02:11<00:57,  6.68it/s]

    The radius of curvature is: 1602.236039408268 m 1488.202337185001 m
    

     70%|██████████████████████████████████████████████████████▉                        | 877/1261 [02:11<00:57,  6.67it/s]

    The radius of curvature is: 1543.7017616905614 m 1376.184836814276 m
    

     70%|███████████████████████████████████████████████████████                        | 878/1261 [02:11<00:57,  6.67it/s]

    The radius of curvature is: 1570.430884199905 m 1032.0578320233878 m
    

     70%|███████████████████████████████████████████████████████                        | 879/1261 [02:11<00:57,  6.67it/s]

    The radius of curvature is: 1576.5435552776678 m 1241.7097931475312 m
    

     70%|███████████████████████████████████████████████████████▏                       | 880/1261 [02:11<00:57,  6.67it/s]

    The radius of curvature is: 1519.2016388760242 m 2257.536885991617 m
    

     70%|███████████████████████████████████████████████████████▏                       | 881/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1388.5220585002587 m 1328.0337749861396 m
    

     70%|███████████████████████████████████████████████████████▎                       | 882/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1218.643010357175 m 1343.5887480539113 m
    

     70%|███████████████████████████████████████████████████████▎                       | 883/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1231.0234899238835 m 2810.6484969488297 m
    

     70%|███████████████████████████████████████████████████████▍                       | 884/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1249.7754906064004 m 1870.0628938518532 m
    

     70%|███████████████████████████████████████████████████████▍                       | 885/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1372.727020968415 m 1588.5642257688924 m
    

     70%|███████████████████████████████████████████████████████▌                       | 886/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1545.593157348313 m 1315.5158527667536 m
    

     70%|███████████████████████████████████████████████████████▌                       | 887/1261 [02:12<00:56,  6.67it/s]

    The radius of curvature is: 1623.773995586784 m 1004.5360963553991 m
    

     70%|███████████████████████████████████████████████████████▋                       | 888/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 1918.5485219769344 m 1266.597163327831 m
    

     70%|███████████████████████████████████████████████████████▋                       | 889/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 2233.0427388457206 m 1217.4377053896546 m
    

     71%|███████████████████████████████████████████████████████▊                       | 890/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 3066.6528319666145 m 891.8043606849692 m
    

     71%|███████████████████████████████████████████████████████▊                       | 891/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 4104.628179745498 m 2151.110450066722 m
    

     71%|███████████████████████████████████████████████████████▉                       | 892/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 5397.196087056 m 1812.2034716783112 m
    

     71%|███████████████████████████████████████████████████████▉                       | 893/1261 [02:13<00:55,  6.67it/s]

    The radius of curvature is: 3794.558768761786 m 1119.0559690008938 m
    

     71%|████████████████████████████████████████████████████████                       | 894/1261 [02:14<00:55,  6.67it/s]

    The radius of curvature is: 3868.5805512885436 m 1809.9129270710373 m
    

     71%|████████████████████████████████████████████████████████                       | 895/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 3141.364119918965 m 6671.4552158768665 m
    

     71%|████████████████████████████████████████████████████████▏                      | 896/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 2890.575706989924 m 2753.4968665582956 m
    

     71%|████████████████████████████████████████████████████████▏                      | 897/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 2897.751378223601 m 50239.65522309286 m
    

     71%|████████████████████████████████████████████████████████▎                      | 898/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 2735.653543815157 m 8814.832708811504 m
    

     71%|████████████████████████████████████████████████████████▎                      | 899/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 2570.639434276259 m 1560.5627902779727 m
    

     71%|████████████████████████████████████████████████████████▍                      | 900/1261 [02:14<00:54,  6.67it/s]

    The radius of curvature is: 3332.076618804384 m 3841.412947113985 m
    

     71%|████████████████████████████████████████████████████████▍                      | 901/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 2955.924690812631 m 1475.4222738644264 m
    

     72%|████████████████████████████████████████████████████████▌                      | 902/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 3909.622625039208 m 4856.7601334110095 m
    

     72%|████████████████████████████████████████████████████████▌                      | 903/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 4076.418063361569 m 6122.391188476495 m
    

     72%|████████████████████████████████████████████████████████▋                      | 904/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 3836.339939248883 m 2010.744035698176 m
    

     72%|████████████████████████████████████████████████████████▋                      | 905/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 3244.437334508975 m 1531.4246876032407 m
    

     72%|████████████████████████████████████████████████████████▊                      | 906/1261 [02:15<00:53,  6.67it/s]

    The radius of curvature is: 2892.5918674495465 m 1711.5128410382822 m
    

     72%|████████████████████████████████████████████████████████▊                      | 907/1261 [02:16<00:53,  6.67it/s]

    The radius of curvature is: 2678.306141125442 m 1663.5684209487001 m
    

     72%|████████████████████████████████████████████████████████▉                      | 908/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 2610.964076810758 m 1507.267861173456 m
    

     72%|████████████████████████████████████████████████████████▉                      | 909/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 2585.2816841309645 m 1051.0538390918148 m
    

     72%|█████████████████████████████████████████████████████████                      | 910/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 2141.8609842437786 m 1292.5434308031136 m
    

     72%|█████████████████████████████████████████████████████████                      | 911/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 1993.0262650152047 m 743.657192972035 m
    

     72%|█████████████████████████████████████████████████████████▏                     | 912/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 1451.1214356796834 m 1085.0075489958015 m
    

     72%|█████████████████████████████████████████████████████████▏                     | 913/1261 [02:16<00:52,  6.67it/s]

    The radius of curvature is: 1435.9431312858646 m 1092.1981588314277 m
    

     72%|█████████████████████████████████████████████████████████▎                     | 914/1261 [02:17<00:52,  6.67it/s]

    The radius of curvature is: 1409.4137586016684 m 1534.2482806009748 m
    

     73%|█████████████████████████████████████████████████████████▎                     | 915/1261 [02:17<00:51,  6.67it/s]

    The radius of curvature is: 1399.7766385531888 m 1131.3614180410675 m
    

     73%|█████████████████████████████████████████████████████████▍                     | 916/1261 [02:17<00:51,  6.67it/s]

    The radius of curvature is: 1141.5150755846644 m 1035.0514510610174 m
    

     73%|█████████████████████████████████████████████████████████▍                     | 917/1261 [02:17<00:51,  6.67it/s]

    The radius of curvature is: 1220.4216101860563 m 4386.7103512889935 m
    

     73%|█████████████████████████████████████████████████████████▌                     | 918/1261 [02:17<00:51,  6.66it/s]

    The radius of curvature is: 1281.5425178359226 m 5697.9209292278465 m
    

     73%|█████████████████████████████████████████████████████████▌                     | 919/1261 [02:17<00:51,  6.66it/s]

    The radius of curvature is: 1093.4667655146268 m 11857.569226167694 m
    

     73%|█████████████████████████████████████████████████████████▋                     | 920/1261 [02:18<00:51,  6.66it/s]

    The radius of curvature is: 1078.936918978171 m 2030.8973170535196 m
    

     73%|█████████████████████████████████████████████████████████▋                     | 921/1261 [02:18<00:51,  6.66it/s]

    The radius of curvature is: 1138.0579246440946 m 1763.369167834671 m
    

     73%|█████████████████████████████████████████████████████████▊                     | 922/1261 [02:18<00:50,  6.66it/s]

    The radius of curvature is: 1180.121710603484 m 1322.5896643468673 m
    

     73%|█████████████████████████████████████████████████████████▊                     | 923/1261 [02:18<00:50,  6.66it/s]

    The radius of curvature is: 1164.1583267445762 m 1098.6755516416022 m
    

     73%|█████████████████████████████████████████████████████████▉                     | 924/1261 [02:18<00:50,  6.66it/s]

    The radius of curvature is: 1261.6247081925767 m 1487.78873385999 m
    

     73%|█████████████████████████████████████████████████████████▉                     | 925/1261 [02:18<00:50,  6.66it/s]

    The radius of curvature is: 1264.4967469109474 m 3798.590931107792 m
    

     73%|██████████████████████████████████████████████████████████                     | 926/1261 [02:18<00:50,  6.66it/s]

    The radius of curvature is: 1643.82413220904 m 1790.6037631010574 m
    

     74%|██████████████████████████████████████████████████████████                     | 927/1261 [02:19<00:50,  6.66it/s]

    The radius of curvature is: 1656.6083439934332 m 1479.1699247477786 m
    

     74%|██████████████████████████████████████████████████████████▏                    | 928/1261 [02:19<00:49,  6.66it/s]

    The radius of curvature is: 1568.1807913129082 m 1737.92650699202 m
    

     74%|██████████████████████████████████████████████████████████▏                    | 929/1261 [02:19<00:49,  6.66it/s]

    The radius of curvature is: 1481.7922629856573 m 5894.310134923529 m
    

     74%|██████████████████████████████████████████████████████████▎                    | 930/1261 [02:19<00:49,  6.66it/s]

    The radius of curvature is: 1618.973544472 m 6571.54048494179 m
    

     74%|██████████████████████████████████████████████████████████▎                    | 931/1261 [02:19<00:49,  6.66it/s]

    The radius of curvature is: 1687.0051637651595 m 3432.2466744363796 m
    

     74%|██████████████████████████████████████████████████████████▍                    | 932/1261 [02:19<00:49,  6.66it/s]

    The radius of curvature is: 1759.66044844624 m 4494.388662711261 m
    

     74%|██████████████████████████████████████████████████████████▍                    | 933/1261 [02:20<00:49,  6.66it/s]

    The radius of curvature is: 2225.5442458421635 m 1540.6440157399725 m
    

     74%|██████████████████████████████████████████████████████████▌                    | 934/1261 [02:20<00:49,  6.66it/s]

    The radius of curvature is: 2648.9336036700283 m 1140.0460016422142 m
    

     74%|██████████████████████████████████████████████████████████▌                    | 935/1261 [02:20<00:48,  6.66it/s]

    The radius of curvature is: 2802.8626062406547 m 1571.199858317813 m
    

     74%|██████████████████████████████████████████████████████████▋                    | 936/1261 [02:20<00:48,  6.66it/s]

    The radius of curvature is: 2593.527914028018 m 836.1557647329938 m
    

     74%|██████████████████████████████████████████████████████████▋                    | 937/1261 [02:20<00:48,  6.66it/s]

    The radius of curvature is: 2819.4711272480986 m 1232.6246255039555 m
    

     74%|██████████████████████████████████████████████████████████▊                    | 938/1261 [02:20<00:48,  6.66it/s]

    The radius of curvature is: 2985.749944575509 m 1091.9119234364625 m
    

     74%|██████████████████████████████████████████████████████████▊                    | 939/1261 [02:20<00:48,  6.66it/s]

    The radius of curvature is: 3108.5549011967555 m 1031.918866053417 m
    

     75%|██████████████████████████████████████████████████████████▉                    | 940/1261 [02:21<00:48,  6.66it/s]

    The radius of curvature is: 1886.8097133097308 m 1179.4555174669292 m
    

     75%|██████████████████████████████████████████████████████████▉                    | 941/1261 [02:21<00:48,  6.66it/s]

    The radius of curvature is: 1681.9929258365275 m 2230.918159410467 m
    

     75%|███████████████████████████████████████████████████████████                    | 942/1261 [02:21<00:47,  6.66it/s]

    The radius of curvature is: 1343.3274500088012 m 1321.6607622010524 m
    

     75%|███████████████████████████████████████████████████████████                    | 943/1261 [02:21<00:47,  6.66it/s]

    The radius of curvature is: 1212.4199707514024 m 3847.2807050364304 m
    

     75%|███████████████████████████████████████████████████████████▏                   | 944/1261 [02:21<00:47,  6.66it/s]

    The radius of curvature is: 1188.1402661960815 m 1201.2881774978562 m
    

     75%|███████████████████████████████████████████████████████████▏                   | 945/1261 [02:21<00:47,  6.66it/s]

    The radius of curvature is: 1048.3556482653692 m 1173.6348100577213 m
    

     75%|███████████████████████████████████████████████████████████▎                   | 946/1261 [02:22<00:47,  6.66it/s]

    The radius of curvature is: 1104.4863082832444 m 1023.5356678220477 m
    

     75%|███████████████████████████████████████████████████████████▎                   | 947/1261 [02:22<00:47,  6.66it/s]

    The radius of curvature is: 1106.6166985103482 m 723.9273864178865 m
    

     75%|███████████████████████████████████████████████████████████▍                   | 948/1261 [02:22<00:46,  6.66it/s]

    The radius of curvature is: 1226.7399982424772 m 698.2544266440761 m
    

     75%|███████████████████████████████████████████████████████████▍                   | 949/1261 [02:22<00:46,  6.66it/s]

    The radius of curvature is: 1370.183942752584 m 1559.5541683497363 m
    

     75%|███████████████████████████████████████████████████████████▌                   | 950/1261 [02:22<00:46,  6.66it/s]

    The radius of curvature is: 1493.0971894516817 m 1571.7083691282794 m
    

     75%|███████████████████████████████████████████████████████████▌                   | 951/1261 [02:22<00:46,  6.66it/s]

    The radius of curvature is: 1862.0226774643113 m 1546.3777672588799 m
    

     75%|███████████████████████████████████████████████████████████▋                   | 952/1261 [02:22<00:46,  6.66it/s]

    The radius of curvature is: 1468.3154413023242 m 1343.9358901699795 m
    

     76%|███████████████████████████████████████████████████████████▋                   | 953/1261 [02:23<00:46,  6.66it/s]

    The radius of curvature is: 1739.9850257305523 m 1496.1113906249525 m
    

     76%|███████████████████████████████████████████████████████████▊                   | 954/1261 [02:23<00:46,  6.66it/s]

    The radius of curvature is: 1664.9852218898225 m 4603.661202337913 m
    

     76%|███████████████████████████████████████████████████████████▊                   | 955/1261 [02:23<00:45,  6.66it/s]

    The radius of curvature is: 1577.8288075367268 m 2428.942134914828 m
    

     76%|███████████████████████████████████████████████████████████▉                   | 956/1261 [02:23<00:45,  6.66it/s]

    The radius of curvature is: 1597.093514416909 m 2422.6101661195626 m
    

     76%|███████████████████████████████████████████████████████████▉                   | 957/1261 [02:23<00:45,  6.66it/s]

    The radius of curvature is: 1826.4844490179303 m 861.4288224188606 m
    

     76%|████████████████████████████████████████████████████████████                   | 958/1261 [02:23<00:45,  6.66it/s]

    The radius of curvature is: 1929.3329759259889 m 1351.0670287069454 m
    

     76%|████████████████████████████████████████████████████████████                   | 959/1261 [02:24<00:45,  6.66it/s]

    The radius of curvature is: 1831.5296860689593 m 1386.2811322983048 m
    

     76%|████████████████████████████████████████████████████████████▏                  | 960/1261 [02:24<00:45,  6.66it/s]

    The radius of curvature is: 1849.7672630958316 m 9752.152881373888 m
    

     76%|████████████████████████████████████████████████████████████▏                  | 961/1261 [02:24<00:45,  6.66it/s]

    The radius of curvature is: 1828.4609251362892 m 1990.5014081739807 m
    

     76%|████████████████████████████████████████████████████████████▎                  | 962/1261 [02:24<00:44,  6.66it/s]

    The radius of curvature is: 1751.3720936167504 m 1623.02069780743 m
    

     76%|████████████████████████████████████████████████████████████▎                  | 963/1261 [02:24<00:44,  6.66it/s]

    The radius of curvature is: 2516.570745986669 m 1383.6719202041568 m
    

     76%|████████████████████████████████████████████████████████████▍                  | 964/1261 [02:24<00:44,  6.66it/s]

    The radius of curvature is: 1715.6224853038063 m 3733.857675121892 m
    

     77%|████████████████████████████████████████████████████████████▍                  | 965/1261 [02:24<00:44,  6.66it/s]

    The radius of curvature is: 1821.5950561817576 m 652242.4748063927 m
    

     77%|████████████████████████████████████████████████████████████▌                  | 966/1261 [02:25<00:44,  6.66it/s]

    The radius of curvature is: 1855.4375612071783 m 2280.640674093993 m
    

     77%|████████████████████████████████████████████████████████████▌                  | 967/1261 [02:25<00:44,  6.66it/s]

    The radius of curvature is: 2166.6186270067346 m 2243.2048369864174 m
    

     77%|████████████████████████████████████████████████████████████▋                  | 968/1261 [02:25<00:44,  6.66it/s]

    The radius of curvature is: 1961.2943034724256 m 1669.9472260055566 m
    

     77%|████████████████████████████████████████████████████████████▋                  | 969/1261 [02:25<00:43,  6.66it/s]

    The radius of curvature is: 1934.3728365682223 m 1086.487310295426 m
    

     77%|████████████████████████████████████████████████████████████▊                  | 970/1261 [02:25<00:43,  6.66it/s]

    The radius of curvature is: 2231.019740915181 m 766.5190153320525 m
    

     77%|████████████████████████████████████████████████████████████▊                  | 971/1261 [02:25<00:43,  6.66it/s]

    The radius of curvature is: 2549.690129970066 m 1008.9042141430117 m
    

     77%|████████████████████████████████████████████████████████████▉                  | 972/1261 [02:26<00:43,  6.66it/s]

    The radius of curvature is: 4240.157954944823 m 966.4532198184654 m
    

     77%|████████████████████████████████████████████████████████████▉                  | 973/1261 [02:26<00:43,  6.65it/s]

    The radius of curvature is: 5091.7018457886725 m 2795.394804238838 m
    

     77%|█████████████████████████████████████████████████████████████                  | 974/1261 [02:26<00:43,  6.66it/s]

    The radius of curvature is: 15945.769584705891 m 1378.3364080788865 m
    

     77%|█████████████████████████████████████████████████████████████                  | 975/1261 [02:26<00:42,  6.65it/s]

    The radius of curvature is: 7062.611782846695 m 1511.8323251376225 m
    

     77%|█████████████████████████████████████████████████████████████▏                 | 976/1261 [02:26<00:42,  6.65it/s]

    The radius of curvature is: 3297.831352489398 m 1092.050931263983 m
    

     77%|█████████████████████████████████████████████████████████████▏                 | 977/1261 [02:26<00:42,  6.65it/s]

    The radius of curvature is: 7307.429201647802 m 1520084.6865079692 m
    

     78%|█████████████████████████████████████████████████████████████▎                 | 978/1261 [02:26<00:42,  6.65it/s]

    The radius of curvature is: 7825.0816997110305 m 25436.617622976595 m
    

     78%|█████████████████████████████████████████████████████████████▎                 | 979/1261 [02:27<00:42,  6.65it/s]

    The radius of curvature is: 2335.589356419579 m 2617.4047107018473 m
    

     78%|█████████████████████████████████████████████████████████████▍                 | 980/1261 [02:27<00:42,  6.65it/s]

    The radius of curvature is: 4677.614136883147 m 1967.3237923844968 m
    

     78%|█████████████████████████████████████████████████████████████▍                 | 981/1261 [02:27<00:42,  6.65it/s]

    The radius of curvature is: 2420.9415858200764 m 4221.050078323593 m
    

     78%|█████████████████████████████████████████████████████████████▌                 | 982/1261 [02:27<00:41,  6.65it/s]

    The radius of curvature is: 1663.0189762114894 m 1082.513340301798 m
    

     78%|█████████████████████████████████████████████████████████████▌                 | 983/1261 [02:27<00:41,  6.65it/s]

    The radius of curvature is: 3099.4424628408783 m 587.1781022090846 m
    

     78%|█████████████████████████████████████████████████████████████▋                 | 984/1261 [02:27<00:41,  6.65it/s]

    The radius of curvature is: 3816.758033933924 m 817.6275043146089 m
    

     78%|█████████████████████████████████████████████████████████████▋                 | 985/1261 [02:28<00:41,  6.65it/s]

    The radius of curvature is: 69467.64804280033 m 1189.1780792805134 m
    

     78%|█████████████████████████████████████████████████████████████▊                 | 986/1261 [02:28<00:41,  6.65it/s]

    The radius of curvature is: 7078.426735571841 m 346.6368830749132 m
    

     78%|█████████████████████████████████████████████████████████████▊                 | 987/1261 [02:28<00:41,  6.65it/s]

    The radius of curvature is: 4922.8593595102175 m 1222.892212826201 m
    

     78%|█████████████████████████████████████████████████████████████▉                 | 988/1261 [02:28<00:41,  6.65it/s]

    The radius of curvature is: 17119.574271552243 m 2648.786937153385 m
    

     78%|█████████████████████████████████████████████████████████████▉                 | 989/1261 [02:28<00:40,  6.65it/s]

    The radius of curvature is: 13220.47789386786 m 1265.0945852475147 m
    

     79%|██████████████████████████████████████████████████████████████                 | 990/1261 [02:28<00:40,  6.65it/s]

    The radius of curvature is: 2399.5231587958287 m 382.45175511022006 m
    

     79%|██████████████████████████████████████████████████████████████                 | 991/1261 [02:28<00:40,  6.65it/s]

    The radius of curvature is: 2196.2670962433167 m 107.79319749504394 m
    

     79%|██████████████████████████████████████████████████████████████▏                | 992/1261 [02:29<00:40,  6.65it/s]

    The radius of curvature is: 2228.795255268158 m 797.8907381509431 m
    

     79%|██████████████████████████████████████████████████████████████▏                | 993/1261 [02:29<00:40,  6.65it/s]

    The radius of curvature is: 3193.6805650042706 m 113.79226524850009 m
    

     79%|██████████████████████████████████████████████████████████████▎                | 994/1261 [02:29<00:40,  6.65it/s]

    The radius of curvature is: 1638.601443510209 m 523.0136236215121 m
    

     79%|██████████████████████████████████████████████████████████████▎                | 995/1261 [02:29<00:39,  6.65it/s]

    The radius of curvature is: 1286.6334358171491 m 206.8054161007302 m
    

     79%|██████████████████████████████████████████████████████████████▍                | 996/1261 [02:29<00:39,  6.65it/s]

    The radius of curvature is: 1689.065504974487 m 173.34174550357395 m
    

     79%|██████████████████████████████████████████████████████████████▍                | 997/1261 [02:29<00:39,  6.66it/s]

    The radius of curvature is: 765.0930052937003 m 845.7398727959493 m
    

     79%|██████████████████████████████████████████████████████████████▌                | 998/1261 [02:29<00:39,  6.66it/s]

    The radius of curvature is: 1060.1612635722022 m 1400.1792337207517 m
    

     79%|██████████████████████████████████████████████████████████████▌                | 999/1261 [02:30<00:39,  6.66it/s]

    The radius of curvature is: 1205.8173546263326 m 1419.6682841351906 m
    

     79%|█████████████████████████████████████████████████████████████▊                | 1000/1261 [02:30<00:39,  6.66it/s]

    The radius of curvature is: 1790.564802520635 m 106.11361981439696 m
    

     79%|█████████████████████████████████████████████████████████████▉                | 1001/1261 [02:30<00:39,  6.66it/s]

    The radius of curvature is: 3021.496622591219 m 32.807888791130885 m
    

     79%|█████████████████████████████████████████████████████████████▉                | 1002/1261 [02:30<00:38,  6.66it/s]

    The radius of curvature is: 3082.0029908865895 m 35.446397584978996 m
    

     80%|██████████████████████████████████████████████████████████████                | 1003/1261 [02:30<00:38,  6.66it/s]

    The radius of curvature is: 162.17873509151562 m 4521.613031603833 m
    

     80%|██████████████████████████████████████████████████████████████                | 1004/1261 [02:30<00:38,  6.66it/s]

    The radius of curvature is: 169.00716771485799 m 1055.4428900605287 m
    

     80%|██████████████████████████████████████████████████████████████▏               | 1005/1261 [02:30<00:38,  6.66it/s]

    The radius of curvature is: 2589.883625113654 m 931.2298403586794 m
    

     80%|██████████████████████████████████████████████████████████████▏               | 1006/1261 [02:31<00:38,  6.66it/s]

    The radius of curvature is: 2229.8907325676487 m 2123.645226588608 m
    

     80%|██████████████████████████████████████████████████████████████▎               | 1007/1261 [02:31<00:38,  6.66it/s]

    The radius of curvature is: 860.8211603907558 m 621.3378530803709 m
    

     80%|██████████████████████████████████████████████████████████████▎               | 1008/1261 [02:31<00:38,  6.66it/s]

    The radius of curvature is: 2420.694851138106 m 797.2511158610904 m
    

     80%|██████████████████████████████████████████████████████████████▍               | 1009/1261 [02:31<00:37,  6.66it/s]

    The radius of curvature is: 1270.842414174181 m 650.6454310417174 m
    

     80%|██████████████████████████████████████████████████████████████▍               | 1010/1261 [02:31<00:37,  6.66it/s]

    The radius of curvature is: 349.8805403377495 m 666.4590942883602 m
    

     80%|██████████████████████████████████████████████████████████████▌               | 1011/1261 [02:31<00:37,  6.66it/s]

    The radius of curvature is: 16.457876153766943 m 685.2346508454923 m
    

     80%|██████████████████████████████████████████████████████████████▌               | 1012/1261 [02:32<00:37,  6.66it/s]

    The radius of curvature is: 81.2630339946858 m 647.2092451828797 m
    

     80%|██████████████████████████████████████████████████████████████▋               | 1013/1261 [02:32<00:37,  6.66it/s]

    The radius of curvature is: 57.20198691844042 m 587.4157971964302 m
    

     80%|██████████████████████████████████████████████████████████████▋               | 1014/1261 [02:32<00:37,  6.66it/s]

    The radius of curvature is: 45.11536618905064 m 799.1855100617179 m
    

     80%|██████████████████████████████████████████████████████████████▊               | 1015/1261 [02:32<00:36,  6.66it/s]

    The radius of curvature is: 1286.4569316684058 m 530.7240587459225 m
    

     81%|██████████████████████████████████████████████████████████████▊               | 1016/1261 [02:32<00:36,  6.66it/s]

    The radius of curvature is: 35.284328784830144 m 764.5488242030659 m
    

     81%|██████████████████████████████████████████████████████████████▉               | 1017/1261 [02:32<00:36,  6.66it/s]

    The radius of curvature is: 128.41910379926034 m 1931.8729566981074 m
    

     81%|██████████████████████████████████████████████████████████████▉               | 1018/1261 [02:32<00:36,  6.66it/s]

    The radius of curvature is: 49.995073456700155 m 1393.5270102178697 m
    

     81%|███████████████████████████████████████████████████████████████               | 1019/1261 [02:33<00:36,  6.66it/s]

    The radius of curvature is: 21.05764540369302 m 1227.9672255920739 m
    

     81%|███████████████████████████████████████████████████████████████               | 1020/1261 [02:33<00:36,  6.66it/s]

    The radius of curvature is: 618.5847997268895 m 1011.527186617429 m
    

     81%|███████████████████████████████████████████████████████████████▏              | 1021/1261 [02:33<00:36,  6.66it/s]

    The radius of curvature is: 52.45122389442681 m 1102.1368665912617 m
    

     81%|███████████████████████████████████████████████████████████████▏              | 1022/1261 [02:33<00:35,  6.66it/s]

    The radius of curvature is: 502.2897745671395 m 1405.6812140573998 m
    

     81%|███████████████████████████████████████████████████████████████▎              | 1023/1261 [02:33<00:35,  6.66it/s]

    The radius of curvature is: 463.794229962002 m 1404.852965018854 m
    

     81%|███████████████████████████████████████████████████████████████▎              | 1024/1261 [02:33<00:35,  6.66it/s]

    The radius of curvature is: 243.50333442134038 m 1508.9831911208848 m
    

     81%|███████████████████████████████████████████████████████████████▍              | 1025/1261 [02:33<00:35,  6.66it/s]

    The radius of curvature is: 773.6543879216335 m 1277.3559316893043 m
    

     81%|███████████████████████████████████████████████████████████████▍              | 1026/1261 [02:34<00:35,  6.66it/s]

    The radius of curvature is: 1131.5970122669464 m 2494.799401897871 m
    

     81%|███████████████████████████████████████████████████████████████▌              | 1027/1261 [02:34<00:35,  6.66it/s]

    The radius of curvature is: 4609.221517282157 m 1825.9118369574437 m
    

     82%|███████████████████████████████████████████████████████████████▌              | 1028/1261 [02:34<00:34,  6.66it/s]

    The radius of curvature is: 2830.2568237681903 m 1063.9899305474376 m
    

     82%|███████████████████████████████████████████████████████████████▋              | 1029/1261 [02:34<00:34,  6.66it/s]

    The radius of curvature is: 487.5240307118071 m 3541.4500673811394 m
    

     82%|███████████████████████████████████████████████████████████████▋              | 1030/1261 [02:34<00:34,  6.66it/s]

    The radius of curvature is: 236.54276151272293 m 1986.5298175917144 m
    

     82%|███████████████████████████████████████████████████████████████▊              | 1031/1261 [02:34<00:34,  6.66it/s]

    The radius of curvature is: 217.95535762863233 m 1325.3643899560234 m
    

     82%|███████████████████████████████████████████████████████████████▊              | 1032/1261 [02:35<00:34,  6.66it/s]

    The radius of curvature is: 313.3483098612304 m 12390.532701605922 m
    

     82%|███████████████████████████████████████████████████████████████▉              | 1033/1261 [02:35<00:34,  6.66it/s]

    The radius of curvature is: 2973.099272425663 m 30355.27791701264 m
    

     82%|███████████████████████████████████████████████████████████████▉              | 1034/1261 [02:35<00:34,  6.66it/s]

    The radius of curvature is: 5257.531241664997 m 1012.935135489464 m
    

     82%|████████████████████████████████████████████████████████████████              | 1035/1261 [02:35<00:33,  6.66it/s]

    The radius of curvature is: 849.6416139528388 m 1130.0501579049999 m
    

     82%|████████████████████████████████████████████████████████████████              | 1036/1261 [02:35<00:33,  6.66it/s]

    The radius of curvature is: 575.0216605210605 m 746.4316712564816 m
    

     82%|████████████████████████████████████████████████████████████████▏             | 1037/1261 [02:35<00:33,  6.66it/s]

    The radius of curvature is: 712.9544691005984 m 548.1571567780189 m
    

     82%|████████████████████████████████████████████████████████████████▏             | 1038/1261 [02:35<00:33,  6.66it/s]

    The radius of curvature is: 850.9888946371552 m 424.61340731101404 m
    

     82%|████████████████████████████████████████████████████████████████▎             | 1039/1261 [02:36<00:33,  6.66it/s]

    The radius of curvature is: 913.9977514173916 m 1470.3672246128262 m
    

     82%|████████████████████████████████████████████████████████████████▎             | 1040/1261 [02:36<00:33,  6.66it/s]

    The radius of curvature is: 978.8594737697149 m 1326.6516293103525 m
    

     83%|████████████████████████████████████████████████████████████████▍             | 1041/1261 [02:36<00:33,  6.66it/s]

    The radius of curvature is: 8623.696496302511 m 24967.974170461548 m
    

     83%|████████████████████████████████████████████████████████████████▍             | 1042/1261 [02:36<00:32,  6.66it/s]

    The radius of curvature is: 2935.886790878895 m 515.0872489739602 m
    

     83%|████████████████████████████████████████████████████████████████▌             | 1043/1261 [02:36<00:32,  6.66it/s]

    The radius of curvature is: 1215.6939983156533 m 497.9342677778107 m
    

     83%|████████████████████████████████████████████████████████████████▌             | 1044/1261 [02:36<00:32,  6.66it/s]

    The radius of curvature is: 679.7948993256714 m 1065.5800371870916 m
    

     83%|████████████████████████████████████████████████████████████████▋             | 1045/1261 [02:37<00:32,  6.66it/s]

    The radius of curvature is: 425.0252118636124 m 1505.8170186705775 m
    

     83%|████████████████████████████████████████████████████████████████▋             | 1046/1261 [02:37<00:32,  6.66it/s]

    The radius of curvature is: 413.3814770589742 m 1190.8362707354543 m
    

     83%|████████████████████████████████████████████████████████████████▊             | 1047/1261 [02:37<00:32,  6.66it/s]

    The radius of curvature is: 593.4814688337741 m 1520.8603155225242 m
    

     83%|████████████████████████████████████████████████████████████████▊             | 1048/1261 [02:37<00:32,  6.66it/s]

    The radius of curvature is: 592.7640561467495 m 1245.1113086054888 m
    

     83%|████████████████████████████████████████████████████████████████▉             | 1049/1261 [02:37<00:31,  6.66it/s]

    The radius of curvature is: 533.1773853150576 m 1283.260740423926 m
    

     83%|████████████████████████████████████████████████████████████████▉             | 1050/1261 [02:37<00:31,  6.66it/s]

    The radius of curvature is: 788.334060552618 m 4043.1411302339284 m
    

     83%|█████████████████████████████████████████████████████████████████             | 1051/1261 [02:37<00:31,  6.66it/s]

    The radius of curvature is: 793.9927944370314 m 1459.0824761663416 m
    

     83%|█████████████████████████████████████████████████████████████████             | 1052/1261 [02:38<00:31,  6.66it/s]

    The radius of curvature is: 1558.8117858258734 m 1063.9255826916685 m
    

     84%|█████████████████████████████████████████████████████████████████▏            | 1053/1261 [02:38<00:31,  6.66it/s]

    The radius of curvature is: 2241.642027754027 m 1059.6857414683677 m
    

     84%|█████████████████████████████████████████████████████████████████▏            | 1054/1261 [02:38<00:31,  6.66it/s]

    The radius of curvature is: 2922.285036830379 m 795.0043057794401 m
    

     84%|█████████████████████████████████████████████████████████████████▎            | 1055/1261 [02:38<00:30,  6.66it/s]

    The radius of curvature is: 4015.4882181083 m 4966.6172654727725 m
    

     84%|█████████████████████████████████████████████████████████████████▎            | 1056/1261 [02:38<00:30,  6.66it/s]

    The radius of curvature is: 4432.3187385604415 m 1300.262728394272 m
    

     84%|█████████████████████████████████████████████████████████████████▍            | 1057/1261 [02:38<00:30,  6.66it/s]

    The radius of curvature is: 4276.9588000435115 m 1032.654603622514 m
    

     84%|█████████████████████████████████████████████████████████████████▍            | 1058/1261 [02:38<00:30,  6.66it/s]

    The radius of curvature is: 3133.437290357824 m 2829.9030783744424 m
    

     84%|█████████████████████████████████████████████████████████████████▌            | 1059/1261 [02:39<00:30,  6.66it/s]

    The radius of curvature is: 2267.1010970407942 m 1426.0742602657303 m
    

     84%|█████████████████████████████████████████████████████████████████▌            | 1060/1261 [02:39<00:30,  6.66it/s]

    The radius of curvature is: 1851.1025337825943 m 801.540328169515 m
    

     84%|█████████████████████████████████████████████████████████████████▋            | 1061/1261 [02:39<00:30,  6.66it/s]

    The radius of curvature is: 1662.982596323578 m 658.1014313309175 m
    

     84%|█████████████████████████████████████████████████████████████████▋            | 1062/1261 [02:39<00:29,  6.65it/s]

    The radius of curvature is: 1611.3447820779702 m 1127.8685856636632 m
    

     84%|█████████████████████████████████████████████████████████████████▊            | 1063/1261 [02:39<00:29,  6.65it/s]

    The radius of curvature is: 1505.3887771536404 m 1084.0934304162815 m
    

     84%|█████████████████████████████████████████████████████████████████▊            | 1064/1261 [02:39<00:29,  6.65it/s]

    The radius of curvature is: 1866.0431426085465 m 1037.9365447616867 m
    

     84%|█████████████████████████████████████████████████████████████████▉            | 1065/1261 [02:40<00:29,  6.65it/s]

    The radius of curvature is: 2113.7266222294134 m 861.620612834596 m
    

     85%|█████████████████████████████████████████████████████████████████▉            | 1066/1261 [02:40<00:29,  6.65it/s]

    The radius of curvature is: 2058.794514174011 m 1064.697421750201 m
    

     85%|██████████████████████████████████████████████████████████████████            | 1067/1261 [02:40<00:29,  6.65it/s]

    The radius of curvature is: 2424.636470564262 m 894.6405949283696 m
    

     85%|██████████████████████████████████████████████████████████████████            | 1068/1261 [02:40<00:29,  6.65it/s]

    The radius of curvature is: 2490.8815491005803 m 826.872644813388 m
    

     85%|██████████████████████████████████████████████████████████████████            | 1069/1261 [02:40<00:28,  6.65it/s]

    The radius of curvature is: 2320.3591041783775 m 1135.439302707485 m
    

     85%|██████████████████████████████████████████████████████████████████▏           | 1070/1261 [02:40<00:28,  6.65it/s]

    The radius of curvature is: 2866.5264660028515 m 825.0668593798254 m
    

     85%|██████████████████████████████████████████████████████████████████▏           | 1071/1261 [02:40<00:28,  6.65it/s]

    The radius of curvature is: 3201.1050554756766 m 1241.7590265466856 m
    

     85%|██████████████████████████████████████████████████████████████████▎           | 1072/1261 [02:41<00:28,  6.65it/s]

    The radius of curvature is: 3254.8330844994857 m 659.8798973135919 m
    

     85%|██████████████████████████████████████████████████████████████████▎           | 1073/1261 [02:41<00:28,  6.65it/s]

    The radius of curvature is: 3742.2639073916735 m 5315.190789591668 m
    

     85%|██████████████████████████████████████████████████████████████████▍           | 1074/1261 [02:41<00:28,  6.65it/s]

    The radius of curvature is: 3976.2951593066537 m 2064.044119475655 m
    

     85%|██████████████████████████████████████████████████████████████████▍           | 1075/1261 [02:41<00:27,  6.65it/s]

    The radius of curvature is: 3449.001262804696 m 1060.5538883504125 m
    

     85%|██████████████████████████████████████████████████████████████████▌           | 1076/1261 [02:41<00:27,  6.65it/s]

    The radius of curvature is: 2405.0048659816457 m 661.0862995902518 m
    

     85%|██████████████████████████████████████████████████████████████████▌           | 1077/1261 [02:41<00:27,  6.65it/s]

    The radius of curvature is: 2101.675254883027 m 1098.4140480783085 m
    

     85%|██████████████████████████████████████████████████████████████████▋           | 1078/1261 [02:42<00:27,  6.65it/s]

    The radius of curvature is: 2316.988286525 m 929.302461360332 m
    

     86%|██████████████████████████████████████████████████████████████████▋           | 1079/1261 [02:42<00:27,  6.65it/s]

    The radius of curvature is: 2098.936986985447 m 1010.6331165716542 m
    

     86%|██████████████████████████████████████████████████████████████████▊           | 1080/1261 [02:42<00:27,  6.65it/s]

    The radius of curvature is: 1973.7514236955228 m 4949.352805654991 m
    

     86%|██████████████████████████████████████████████████████████████████▊           | 1081/1261 [02:42<00:27,  6.65it/s]

    The radius of curvature is: 1856.8083363726332 m 1834.480186303006 m
    

     86%|██████████████████████████████████████████████████████████████████▉           | 1082/1261 [02:42<00:26,  6.65it/s]

    The radius of curvature is: 2073.0576145087052 m 1038.5586103853677 m
    

     86%|██████████████████████████████████████████████████████████████████▉           | 1083/1261 [02:42<00:26,  6.65it/s]

    The radius of curvature is: 1948.0589474014 m 646.8851536929308 m
    

     86%|███████████████████████████████████████████████████████████████████           | 1084/1261 [02:42<00:26,  6.65it/s]

    The radius of curvature is: 1789.7375532653116 m 919.5975066289043 m
    

     86%|███████████████████████████████████████████████████████████████████           | 1085/1261 [02:43<00:26,  6.65it/s]

    The radius of curvature is: 2165.7098291424554 m 2258.662052475525 m
    

     86%|███████████████████████████████████████████████████████████████████▏          | 1086/1261 [02:43<00:26,  6.65it/s]

    The radius of curvature is: 2012.3240477869065 m 1995.5339912950005 m
    

     86%|███████████████████████████████████████████████████████████████████▏          | 1087/1261 [02:43<00:26,  6.65it/s]

    The radius of curvature is: 2323.3497914520963 m 2378.0681660871746 m
    

     86%|███████████████████████████████████████████████████████████████████▎          | 1088/1261 [02:43<00:26,  6.65it/s]

    The radius of curvature is: 2279.157857163797 m 954.2915539735726 m
    

     86%|███████████████████████████████████████████████████████████████████▎          | 1089/1261 [02:43<00:25,  6.65it/s]

    The radius of curvature is: 2070.7035221206856 m 628.2887083716128 m
    

     86%|███████████████████████████████████████████████████████████████████▍          | 1090/1261 [02:43<00:25,  6.65it/s]

    The radius of curvature is: 2638.203024776525 m 868.2305512289872 m
    

     87%|███████████████████████████████████████████████████████████████████▍          | 1091/1261 [02:44<00:25,  6.65it/s]

    The radius of curvature is: 2287.6530777182625 m 712.4477606657407 m
    

     87%|███████████████████████████████████████████████████████████████████▌          | 1092/1261 [02:44<00:25,  6.65it/s]

    The radius of curvature is: 1820.6926203804019 m 1327.2224029923989 m
    

     87%|███████████████████████████████████████████████████████████████████▌          | 1093/1261 [02:44<00:25,  6.65it/s]

    The radius of curvature is: 1859.7613047071065 m 1327.4487102761004 m
    

     87%|███████████████████████████████████████████████████████████████████▋          | 1094/1261 [02:44<00:25,  6.65it/s]

    The radius of curvature is: 1950.3042020947041 m 1260.790123406287 m
    

     87%|███████████████████████████████████████████████████████████████████▋          | 1095/1261 [02:44<00:24,  6.65it/s]

    The radius of curvature is: 2014.1375192369292 m 1084.5667101527022 m
    

     87%|███████████████████████████████████████████████████████████████████▊          | 1096/1261 [02:44<00:24,  6.65it/s]

    The radius of curvature is: 2335.748984157223 m 1504.6676953303427 m
    

     87%|███████████████████████████████████████████████████████████████████▊          | 1097/1261 [02:44<00:24,  6.65it/s]

    The radius of curvature is: 2899.2307743725323 m 1498.1247901059392 m
    

     87%|███████████████████████████████████████████████████████████████████▉          | 1098/1261 [02:45<00:24,  6.65it/s]

    The radius of curvature is: 4510.513004547358 m 2496.416210809622 m
    

     87%|███████████████████████████████████████████████████████████████████▉          | 1099/1261 [02:45<00:24,  6.65it/s]

    The radius of curvature is: 37761.49033095383 m 2304.4494921128917 m
    

     87%|████████████████████████████████████████████████████████████████████          | 1100/1261 [02:45<00:24,  6.65it/s]

    The radius of curvature is: 16967.64621047094 m 1108.444600129579 m
    

     87%|████████████████████████████████████████████████████████████████████          | 1101/1261 [02:45<00:24,  6.65it/s]

    The radius of curvature is: 12250.479761936398 m 822.8513222885936 m
    

     87%|████████████████████████████████████████████████████████████████████▏         | 1102/1261 [02:45<00:23,  6.65it/s]

    The radius of curvature is: 9151.1369440623 m 956.7062797362992 m
    

     87%|████████████████████████████████████████████████████████████████████▏         | 1103/1261 [02:45<00:23,  6.65it/s]

    The radius of curvature is: 23387.437545559147 m 1259.524797661404 m
    

     88%|████████████████████████████████████████████████████████████████████▎         | 1104/1261 [02:45<00:23,  6.65it/s]

    The radius of curvature is: 23813.487508840473 m 5955.285937979292 m
    

     88%|████████████████████████████████████████████████████████████████████▎         | 1105/1261 [02:46<00:23,  6.65it/s]

    The radius of curvature is: 5175.651273949967 m 1600.4302484715372 m
    

     88%|████████████████████████████████████████████████████████████████████▍         | 1106/1261 [02:46<00:23,  6.65it/s]

    The radius of curvature is: 3188.0143828118976 m 1047.255035889794 m
    

     88%|████████████████████████████████████████████████████████████████████▍         | 1107/1261 [02:46<00:23,  6.65it/s]

    The radius of curvature is: 2108.736592696168 m 2460.6441395480124 m
    

     88%|████████████████████████████████████████████████████████████████████▌         | 1108/1261 [02:46<00:23,  6.65it/s]

    The radius of curvature is: 2303.1889605953465 m 1730.937815851882 m
    

     88%|████████████████████████████████████████████████████████████████████▌         | 1109/1261 [02:46<00:22,  6.65it/s]

    The radius of curvature is: 2026.6848840605865 m 2881.637843048486 m
    

     88%|████████████████████████████████████████████████████████████████████▋         | 1110/1261 [02:46<00:22,  6.65it/s]

    The radius of curvature is: 1694.0222247743486 m 1312.764442241534 m
    

     88%|████████████████████████████████████████████████████████████████████▋         | 1111/1261 [02:47<00:22,  6.65it/s]

    The radius of curvature is: 1707.7484567436375 m 948.7430020935113 m
    

     88%|████████████████████████████████████████████████████████████████████▊         | 1112/1261 [02:47<00:22,  6.65it/s]

    The radius of curvature is: 1676.8258912282106 m 673.4821739797515 m
    

     88%|████████████████████████████████████████████████████████████████████▊         | 1113/1261 [02:47<00:22,  6.65it/s]

    The radius of curvature is: 2399.0549213961353 m 608.6707365239249 m
    

     88%|████████████████████████████████████████████████████████████████████▉         | 1114/1261 [02:47<00:22,  6.65it/s]

    The radius of curvature is: 2231.231254623459 m 698.5321375719474 m
    

     88%|████████████████████████████████████████████████████████████████████▉         | 1115/1261 [02:47<00:21,  6.65it/s]

    The radius of curvature is: 2834.543779845482 m 815.0027654684733 m
    

     89%|█████████████████████████████████████████████████████████████████████         | 1116/1261 [02:47<00:21,  6.65it/s]

    The radius of curvature is: 3813.3210784511034 m 751.8758203670299 m
    

     89%|█████████████████████████████████████████████████████████████████████         | 1117/1261 [02:47<00:21,  6.65it/s]

    The radius of curvature is: 4625.397776933877 m 1258.6891660665506 m
    

     89%|█████████████████████████████████████████████████████████████████████▏        | 1118/1261 [02:48<00:21,  6.65it/s]

    The radius of curvature is: 6342.0023451001025 m 934.3355187191275 m
    

     89%|█████████████████████████████████████████████████████████████████████▏        | 1119/1261 [02:48<00:21,  6.65it/s]

    The radius of curvature is: 8266.560823984599 m 2328.475572040159 m
    

     89%|█████████████████████████████████████████████████████████████████████▎        | 1120/1261 [02:48<00:21,  6.65it/s]

    The radius of curvature is: 14987.851858359236 m 1608.5109235558575 m
    

     89%|█████████████████████████████████████████████████████████████████████▎        | 1121/1261 [02:48<00:21,  6.65it/s]

    The radius of curvature is: 21921.874496749853 m 1001.8319259140802 m
    

     89%|█████████████████████████████████████████████████████████████████████▍        | 1122/1261 [02:48<00:20,  6.65it/s]

    The radius of curvature is: 28119.581131231193 m 767.0973275233246 m
    

     89%|█████████████████████████████████████████████████████████████████████▍        | 1123/1261 [02:48<00:20,  6.65it/s]

    The radius of curvature is: 78701.71630235779 m 777.7597845663103 m
    

     89%|█████████████████████████████████████████████████████████████████████▌        | 1124/1261 [02:48<00:20,  6.65it/s]

    The radius of curvature is: 156963.24938572507 m 1231.71921806693 m
    

     89%|█████████████████████████████████████████████████████████████████████▌        | 1125/1261 [02:49<00:20,  6.65it/s]

    The radius of curvature is: 17051.422523927664 m 1021.5571185690417 m
    

     89%|█████████████████████████████████████████████████████████████████████▋        | 1126/1261 [02:49<00:20,  6.65it/s]

    The radius of curvature is: 4142.523138766175 m 847.6265847270286 m
    

     89%|█████████████████████████████████████████████████████████████████████▋        | 1127/1261 [02:49<00:20,  6.65it/s]

    The radius of curvature is: 3030.4278657301124 m 1499.1070349126262 m
    

     89%|█████████████████████████████████████████████████████████████████████▊        | 1128/1261 [02:49<00:19,  6.65it/s]

    The radius of curvature is: 1947.2333251062955 m 951.4402019812202 m
    

     90%|█████████████████████████████████████████████████████████████████████▊        | 1129/1261 [02:49<00:19,  6.65it/s]

    The radius of curvature is: 1601.1681044720276 m 1080.2614196404509 m
    

     90%|█████████████████████████████████████████████████████████████████████▉        | 1130/1261 [02:49<00:19,  6.65it/s]

    The radius of curvature is: 1503.0513739184041 m 42013.8035628685 m
    

     90%|█████████████████████████████████████████████████████████████████████▉        | 1131/1261 [02:50<00:19,  6.65it/s]

    The radius of curvature is: 1223.0642820109563 m 1765.7134082563757 m
    

     90%|██████████████████████████████████████████████████████████████████████        | 1132/1261 [02:50<00:19,  6.65it/s]

    The radius of curvature is: 1531.0843848633383 m 1506.3955614772926 m
    

     90%|██████████████████████████████████████████████████████████████████████        | 1133/1261 [02:50<00:19,  6.65it/s]

    The radius of curvature is: 1647.6171811173915 m 940.233026460118 m
    

     90%|██████████████████████████████████████████████████████████████████████▏       | 1134/1261 [02:50<00:19,  6.65it/s]

    The radius of curvature is: 1876.5477363846023 m 729.0940666894776 m
    

     90%|██████████████████████████████████████████████████████████████████████▏       | 1135/1261 [02:50<00:18,  6.65it/s]

    The radius of curvature is: 2137.662038551832 m 804.3961311872777 m
    

     90%|██████████████████████████████████████████████████████████████████████▎       | 1136/1261 [02:50<00:18,  6.65it/s]

    The radius of curvature is: 2879.4751680342197 m 1808.7179996913658 m
    

     90%|██████████████████████████████████████████████████████████████████████▎       | 1137/1261 [02:50<00:18,  6.65it/s]

    The radius of curvature is: 3594.9984831074626 m 1690.6227206563894 m
    

     90%|██████████████████████████████████████████████████████████████████████▍       | 1138/1261 [02:51<00:18,  6.65it/s]

    The radius of curvature is: 3272.4983362917646 m 1207.0821014524852 m
    

     90%|██████████████████████████████████████████████████████████████████████▍       | 1139/1261 [02:51<00:18,  6.65it/s]

    The radius of curvature is: 4165.071871289503 m 838.8747499437111 m
    

     90%|██████████████████████████████████████████████████████████████████████▌       | 1140/1261 [02:51<00:18,  6.65it/s]

    The radius of curvature is: 5611.59362418442 m 1307.5881827062858 m
    

     90%|██████████████████████████████████████████████████████████████████████▌       | 1141/1261 [02:51<00:18,  6.65it/s]

    The radius of curvature is: 5326.92456870563 m 8038.03535589865 m
    

     91%|██████████████████████████████████████████████████████████████████████▋       | 1142/1261 [02:51<00:17,  6.65it/s]

    The radius of curvature is: 3937.6286567787183 m 4045.1560872226614 m
    

     91%|██████████████████████████████████████████████████████████████████████▋       | 1143/1261 [02:51<00:17,  6.65it/s]

    The radius of curvature is: 3824.5329641782687 m 1556.9426201585018 m
    

     91%|██████████████████████████████████████████████████████████████████████▊       | 1144/1261 [02:51<00:17,  6.65it/s]

    The radius of curvature is: 4806.429136340841 m 1809.670393850339 m
    

     91%|██████████████████████████████████████████████████████████████████████▊       | 1145/1261 [02:52<00:17,  6.65it/s]

    The radius of curvature is: 6489.041848398452 m 801.0961092179581 m
    

     91%|██████████████████████████████████████████████████████████████████████▉       | 1146/1261 [02:52<00:17,  6.65it/s]

    The radius of curvature is: 9243.685188745276 m 626.586630726673 m
    

     91%|██████████████████████████████████████████████████████████████████████▉       | 1147/1261 [02:52<00:17,  6.65it/s]

    The radius of curvature is: 11096.742811031932 m 685.3797246120138 m
    

     91%|███████████████████████████████████████████████████████████████████████       | 1148/1261 [02:52<00:16,  6.65it/s]

    The radius of curvature is: 5668.5579489090005 m 695.8737958782023 m
    

     91%|███████████████████████████████████████████████████████████████████████       | 1149/1261 [02:52<00:16,  6.65it/s]

    The radius of curvature is: 3365.9730093435455 m 1389.8156438587014 m
    

     91%|███████████████████████████████████████████████████████████████████████▏      | 1150/1261 [02:52<00:16,  6.65it/s]

    The radius of curvature is: 2332.512267803979 m 1478.0290260327001 m
    

     91%|███████████████████████████████████████████████████████████████████████▏      | 1151/1261 [02:53<00:16,  6.65it/s]

    The radius of curvature is: 2411.450762071548 m 1205.612484737052 m
    

     91%|███████████████████████████████████████████████████████████████████████▎      | 1152/1261 [02:53<00:16,  6.65it/s]

    The radius of curvature is: 2014.873662166398 m 7357.902895734716 m
    

     91%|███████████████████████████████████████████████████████████████████████▎      | 1153/1261 [02:53<00:16,  6.65it/s]

    The radius of curvature is: 2468.522138712333 m 5164.292320253879 m
    

     92%|███████████████████████████████████████████████████████████████████████▍      | 1154/1261 [02:53<00:16,  6.65it/s]

    The radius of curvature is: 2588.0586311000725 m 2467.0107931004786 m
    

     92%|███████████████████████████████████████████████████████████████████████▍      | 1155/1261 [02:53<00:15,  6.65it/s]

    The radius of curvature is: 3268.42500978485 m 1080.1511326291038 m
    

     92%|███████████████████████████████████████████████████████████████████████▌      | 1156/1261 [02:53<00:15,  6.65it/s]

    The radius of curvature is: 3660.179838044189 m 1057.842046480736 m
    

     92%|███████████████████████████████████████████████████████████████████████▌      | 1157/1261 [02:53<00:15,  6.65it/s]

    The radius of curvature is: 4032.8599147222267 m 1000.0912147481935 m
    

     92%|███████████████████████████████████████████████████████████████████████▋      | 1158/1261 [02:54<00:15,  6.65it/s]

    The radius of curvature is: 3743.1725237303863 m 1423.4592589643341 m
    

     92%|███████████████████████████████████████████████████████████████████████▋      | 1159/1261 [02:54<00:15,  6.65it/s]

    The radius of curvature is: 2867.0836964970626 m 741.7788714499849 m
    

     92%|███████████████████████████████████████████████████████████████████████▊      | 1160/1261 [02:54<00:15,  6.65it/s]

    The radius of curvature is: 1910.0364634980256 m 1284.0476293310876 m
    

     92%|███████████████████████████████████████████████████████████████████████▊      | 1161/1261 [02:54<00:15,  6.65it/s]

    The radius of curvature is: 1545.1662439212912 m 1262.6280708545785 m
    

     92%|███████████████████████████████████████████████████████████████████████▉      | 1162/1261 [02:54<00:14,  6.65it/s]

    The radius of curvature is: 1232.8879411752418 m 1135.0705171551092 m
    

     92%|███████████████████████████████████████████████████████████████████████▉      | 1163/1261 [02:54<00:14,  6.65it/s]

    The radius of curvature is: 1157.4978103082003 m 2504.1647062586167 m
    

     92%|████████████████████████████████████████████████████████████████████████      | 1164/1261 [02:54<00:14,  6.65it/s]

    The radius of curvature is: 1054.301935698358 m 2622.135736066825 m
    

     92%|████████████████████████████████████████████████████████████████████████      | 1165/1261 [02:55<00:14,  6.65it/s]

    The radius of curvature is: 1027.5357540449247 m 1508.0507105551417 m
    

     92%|████████████████████████████████████████████████████████████████████████      | 1166/1261 [02:55<00:14,  6.65it/s]

    The radius of curvature is: 1137.8242341219718 m 1259.4831669484151 m
    

     93%|████████████████████████████████████████████████████████████████████████▏     | 1167/1261 [02:55<00:14,  6.65it/s]

    The radius of curvature is: 1340.88436803421 m 981.1097376357668 m
    

     93%|████████████████████████████████████████████████████████████████████████▏     | 1168/1261 [02:55<00:13,  6.65it/s]

    The radius of curvature is: 1428.2013604059027 m 589.7369603201751 m
    

     93%|████████████████████████████████████████████████████████████████████████▎     | 1169/1261 [02:55<00:13,  6.65it/s]

    The radius of curvature is: 1641.5253636140735 m 797.291909741837 m
    

     93%|████████████████████████████████████████████████████████████████████████▎     | 1170/1261 [02:55<00:13,  6.65it/s]

    The radius of curvature is: 1946.685004006206 m 911.7216096173848 m
    

     93%|████████████████████████████████████████████████████████████████████████▍     | 1171/1261 [02:56<00:13,  6.65it/s]

    The radius of curvature is: 2383.6920360226127 m 1005.24123362936 m
    

     93%|████████████████████████████████████████████████████████████████████████▍     | 1172/1261 [02:56<00:13,  6.65it/s]

    The radius of curvature is: 2654.7656195958634 m 1187.7780151489046 m
    

     93%|████████████████████████████████████████████████████████████████████████▌     | 1173/1261 [02:56<00:13,  6.65it/s]

    The radius of curvature is: 2824.5020756517183 m 908.0874195822772 m
    

     93%|████████████████████████████████████████████████████████████████████████▌     | 1174/1261 [02:56<00:13,  6.65it/s]

    The radius of curvature is: 2627.0589033010556 m 896.2236920979149 m
    

     93%|████████████████████████████████████████████████████████████████████████▋     | 1175/1261 [02:56<00:12,  6.65it/s]

    The radius of curvature is: 2611.972947016028 m 2137.763437019044 m
    

     93%|████████████████████████████████████████████████████████████████████████▋     | 1176/1261 [02:56<00:12,  6.65it/s]

    The radius of curvature is: 2691.352416879316 m 2286.4441415800493 m
    

     93%|████████████████████████████████████████████████████████████████████████▊     | 1177/1261 [02:56<00:12,  6.65it/s]

    The radius of curvature is: 2561.917796047027 m 1971.505893544005 m
    

     93%|████████████████████████████████████████████████████████████████████████▊     | 1178/1261 [02:57<00:12,  6.65it/s]

    The radius of curvature is: 3607.393818199878 m 1233.4605130165485 m
    

     93%|████████████████████████████████████████████████████████████████████████▉     | 1179/1261 [02:57<00:12,  6.65it/s]

    The radius of curvature is: 4473.218614158032 m 1369.4781157197829 m
    

     94%|████████████████████████████████████████████████████████████████████████▉     | 1180/1261 [02:57<00:12,  6.65it/s]

    The radius of curvature is: 5813.435345826853 m 2359.6313562342802 m
    

     94%|█████████████████████████████████████████████████████████████████████████     | 1181/1261 [02:57<00:12,  6.65it/s]

    The radius of curvature is: 9231.617362587669 m 1166.8963552605444 m
    

     94%|█████████████████████████████████████████████████████████████████████████     | 1182/1261 [02:57<00:11,  6.65it/s]

    The radius of curvature is: 5230.502064116508 m 1571.9447457824308 m
    

     94%|█████████████████████████████████████████████████████████████████████████▏    | 1183/1261 [02:57<00:11,  6.65it/s]

    The radius of curvature is: 4463.844231239442 m 1470.9882594568212 m
    

     94%|█████████████████████████████████████████████████████████████████████████▏    | 1184/1261 [02:58<00:11,  6.65it/s]

    The radius of curvature is: 2873.2026064174 m 1417.784065647245 m
    

     94%|█████████████████████████████████████████████████████████████████████████▎    | 1185/1261 [02:58<00:11,  6.65it/s]

    The radius of curvature is: 2507.51536607651 m 1821.5036946893938 m
    

     94%|█████████████████████████████████████████████████████████████████████████▎    | 1186/1261 [02:58<00:11,  6.65it/s]

    The radius of curvature is: 2235.8763319651844 m 920.5155406499979 m
    

     94%|█████████████████████████████████████████████████████████████████████████▍    | 1187/1261 [02:58<00:11,  6.65it/s]

    The radius of curvature is: 2281.4561358522315 m 957.841630388049 m
    

     94%|█████████████████████████████████████████████████████████████████████████▍    | 1188/1261 [02:58<00:10,  6.65it/s]

    The radius of curvature is: 2484.100608950571 m 1349.51694318244 m
    

     94%|█████████████████████████████████████████████████████████████████████████▌    | 1189/1261 [02:58<00:10,  6.65it/s]

    The radius of curvature is: 2298.872623592589 m 739.4004802472928 m
    

     94%|█████████████████████████████████████████████████████████████████████████▌    | 1190/1261 [02:58<00:10,  6.65it/s]

    The radius of curvature is: 2238.2315508678507 m 762.3542648201366 m
    

     94%|█████████████████████████████████████████████████████████████████████████▋    | 1191/1261 [02:59<00:10,  6.65it/s]

    The radius of curvature is: 1763.7660911058501 m 699.928464586368 m
    

     95%|█████████████████████████████████████████████████████████████████████████▋    | 1192/1261 [02:59<00:10,  6.65it/s]

    The radius of curvature is: 2098.8761374541778 m 836.4923024344081 m
    

     95%|█████████████████████████████████████████████████████████████████████████▊    | 1193/1261 [02:59<00:10,  6.65it/s]

    The radius of curvature is: 1719.6604565524324 m 1616.0195986239662 m
    

     95%|█████████████████████████████████████████████████████████████████████████▊    | 1194/1261 [02:59<00:10,  6.65it/s]

    The radius of curvature is: 1521.1402326730615 m 1007.7983228666793 m
    

     95%|█████████████████████████████████████████████████████████████████████████▉    | 1195/1261 [02:59<00:09,  6.65it/s]

    The radius of curvature is: 1210.434454174694 m 1047.335565163893 m
    

     95%|█████████████████████████████████████████████████████████████████████████▉    | 1196/1261 [02:59<00:09,  6.65it/s]

    The radius of curvature is: 1138.5473202055675 m 795.6351217196184 m
    

     95%|██████████████████████████████████████████████████████████████████████████    | 1197/1261 [02:59<00:09,  6.65it/s]

    The radius of curvature is: 1300.5545958953578 m 1187.9692902148693 m
    

     95%|██████████████████████████████████████████████████████████████████████████    | 1198/1261 [03:00<00:09,  6.65it/s]

    The radius of curvature is: 1393.6207574175946 m 1328.0917503400694 m
    

     95%|██████████████████████████████████████████████████████████████████████████▏   | 1199/1261 [03:00<00:09,  6.65it/s]

    The radius of curvature is: 1495.5593914287333 m 1739.7328690743982 m
    

     95%|██████████████████████████████████████████████████████████████████████████▏   | 1200/1261 [03:00<00:09,  6.65it/s]

    The radius of curvature is: 1693.5611362852324 m 1087.3871998474817 m
    

     95%|██████████████████████████████████████████████████████████████████████████▎   | 1201/1261 [03:00<00:09,  6.65it/s]

    The radius of curvature is: 3580.339367813752 m 701.3509219397622 m
    

     95%|██████████████████████████████████████████████████████████████████████████▎   | 1202/1261 [03:00<00:08,  6.65it/s]

    The radius of curvature is: 3053.8843229914223 m 816.8007429870617 m
    

     95%|██████████████████████████████████████████████████████████████████████████▍   | 1203/1261 [03:00<00:08,  6.65it/s]

    The radius of curvature is: 3119.5490047573167 m 815.8831636611241 m
    

     95%|██████████████████████████████████████████████████████████████████████████▍   | 1204/1261 [03:01<00:08,  6.65it/s]

    The radius of curvature is: 2742.572968398843 m 907.3151754360083 m
    

     96%|██████████████████████████████████████████████████████████████████████████▌   | 1205/1261 [03:01<00:08,  6.65it/s]

    The radius of curvature is: 1948.5863490513761 m 1129.521416284054 m
    

     96%|██████████████████████████████████████████████████████████████████████████▌   | 1206/1261 [03:01<00:08,  6.65it/s]

    The radius of curvature is: 1307.2706348018176 m 702.8786014230423 m
    

     96%|██████████████████████████████████████████████████████████████████████████▋   | 1207/1261 [03:01<00:08,  6.65it/s]

    The radius of curvature is: 1236.7956484487795 m 587.2581857842247 m
    

     96%|██████████████████████████████████████████████████████████████████████████▋   | 1208/1261 [03:01<00:07,  6.65it/s]

    The radius of curvature is: 1229.5846430757283 m 1278.7919906055665 m
    

     96%|██████████████████████████████████████████████████████████████████████████▊   | 1209/1261 [03:01<00:07,  6.65it/s]

    The radius of curvature is: 1200.6044090185235 m 2368.4246471212086 m
    

     96%|██████████████████████████████████████████████████████████████████████████▊   | 1210/1261 [03:01<00:07,  6.65it/s]

    The radius of curvature is: 1431.222467847712 m 923.4800208708177 m
    

     96%|██████████████████████████████████████████████████████████████████████████▉   | 1211/1261 [03:02<00:07,  6.65it/s]

    The radius of curvature is: 1506.3753182871703 m 886.5647299566448 m
    

     96%|██████████████████████████████████████████████████████████████████████████▉   | 1212/1261 [03:02<00:07,  6.65it/s]

    The radius of curvature is: 1614.5946232125054 m 1011.7093166525608 m
    

     96%|███████████████████████████████████████████████████████████████████████████   | 1213/1261 [03:02<00:07,  6.65it/s]

    The radius of curvature is: 2007.481230836878 m 683.0442642143095 m
    

     96%|███████████████████████████████████████████████████████████████████████████   | 1214/1261 [03:02<00:07,  6.65it/s]

    The radius of curvature is: 1882.1891806704225 m 488.32880977088763 m
    

     96%|███████████████████████████████████████████████████████████████████████████▏  | 1215/1261 [03:02<00:06,  6.65it/s]

    The radius of curvature is: 2643.9906869985502 m 677.0791783503091 m
    

     96%|███████████████████████████████████████████████████████████████████████████▏  | 1216/1261 [03:02<00:06,  6.65it/s]

    The radius of curvature is: 2962.0613693100204 m 1762.328930488912 m
    

     97%|███████████████████████████████████████████████████████████████████████████▎  | 1217/1261 [03:03<00:06,  6.65it/s]

    The radius of curvature is: 2189.162281322377 m 952.5157549104823 m
    

     97%|███████████████████████████████████████████████████████████████████████████▎  | 1218/1261 [03:03<00:06,  6.65it/s]

    The radius of curvature is: 1912.2786672088082 m 963.122816915501 m
    

     97%|███████████████████████████████████████████████████████████████████████████▍  | 1219/1261 [03:03<00:06,  6.65it/s]

    The radius of curvature is: 1871.282258737698 m 696.5430736996967 m
    

     97%|███████████████████████████████████████████████████████████████████████████▍  | 1220/1261 [03:03<00:06,  6.65it/s]

    The radius of curvature is: 1967.5592396594723 m 6667.55280102905 m
    

     97%|███████████████████████████████████████████████████████████████████████████▌  | 1221/1261 [03:03<00:06,  6.65it/s]

    The radius of curvature is: 1677.3454442776224 m 1201.9035351502255 m
    

     97%|███████████████████████████████████████████████████████████████████████████▌  | 1222/1261 [03:03<00:05,  6.65it/s]

    The radius of curvature is: 1688.9165346777395 m 1205.119919682847 m
    

     97%|███████████████████████████████████████████████████████████████████████████▋  | 1223/1261 [03:03<00:05,  6.65it/s]

    The radius of curvature is: 1907.0526184457283 m 666.2997237493558 m
    

     97%|███████████████████████████████████████████████████████████████████████████▋  | 1224/1261 [03:04<00:05,  6.65it/s]

    The radius of curvature is: 2425.3271450112516 m 746.965432498985 m
    

     97%|███████████████████████████████████████████████████████████████████████████▊  | 1225/1261 [03:04<00:05,  6.65it/s]

    The radius of curvature is: 4130.425638095692 m 617.8877283165946 m
    

     97%|███████████████████████████████████████████████████████████████████████████▊  | 1226/1261 [03:04<00:05,  6.65it/s]

    The radius of curvature is: 10283.19581070857 m 920.7623714629798 m
    

     97%|███████████████████████████████████████████████████████████████████████████▉  | 1227/1261 [03:04<00:05,  6.65it/s]

    The radius of curvature is: 12754.698425541432 m 834.8411005208077 m
    

     97%|███████████████████████████████████████████████████████████████████████████▉  | 1228/1261 [03:04<00:04,  6.65it/s]

    The radius of curvature is: 4024.084958109555 m 1146.6423158269436 m
    

     97%|████████████████████████████████████████████████████████████████████████████  | 1229/1261 [03:04<00:04,  6.65it/s]

    The radius of curvature is: 4181.049014109352 m 926.1399083288674 m
    

     98%|████████████████████████████████████████████████████████████████████████████  | 1230/1261 [03:05<00:04,  6.65it/s]

    The radius of curvature is: 2732.6015966151954 m 877.8218308118034 m
    

     98%|████████████████████████████████████████████████████████████████████████████▏ | 1231/1261 [03:05<00:04,  6.65it/s]

    The radius of curvature is: 2465.4092707744985 m 624.6153383794237 m
    

     98%|████████████████████████████████████████████████████████████████████████████▏ | 1232/1261 [03:05<00:04,  6.65it/s]

    The radius of curvature is: 1929.7134365217494 m 2549.7963158684356 m
    

     98%|████████████████████████████████████████████████████████████████████████████▎ | 1233/1261 [03:05<00:04,  6.65it/s]

    The radius of curvature is: 1885.5293403932424 m 1478.3107673888312 m
    

     98%|████████████████████████████████████████████████████████████████████████████▎ | 1234/1261 [03:05<00:04,  6.65it/s]

    The radius of curvature is: 2477.3708448628204 m 2288.6722547955083 m
    

     98%|████████████████████████████████████████████████████████████████████████████▍ | 1235/1261 [03:05<00:03,  6.65it/s]

    The radius of curvature is: 4425.08955818201 m 865.2194627560352 m
    

     98%|████████████████████████████████████████████████████████████████████████████▍ | 1236/1261 [03:05<00:03,  6.65it/s]

    The radius of curvature is: 532956.330627206 m 781.4599673430471 m
    

     98%|████████████████████████████████████████████████████████████████████████████▌ | 1237/1261 [03:06<00:03,  6.65it/s]

    The radius of curvature is: 53170.548092634664 m 1675.964337163926 m
    

     98%|████████████████████████████████████████████████████████████████████████████▌ | 1238/1261 [03:06<00:03,  6.65it/s]

    The radius of curvature is: 29038.72750418769 m 11706.128719435674 m
    

     98%|████████████████████████████████████████████████████████████████████████████▋ | 1239/1261 [03:06<00:03,  6.65it/s]

    The radius of curvature is: 41926.486756090926 m 1691.4881569778004 m
    

     98%|████████████████████████████████████████████████████████████████████████████▋ | 1240/1261 [03:06<00:03,  6.65it/s]

    The radius of curvature is: 22188.356531341986 m 1512.1117497054615 m
    

     98%|████████████████████████████████████████████████████████████████████████████▊ | 1241/1261 [03:06<00:03,  6.65it/s]

    The radius of curvature is: 10224.522160199644 m 1987.8819084155648 m
    

     98%|████████████████████████████████████████████████████████████████████████████▊ | 1242/1261 [03:06<00:02,  6.65it/s]

    The radius of curvature is: 4576.195592647895 m 10420.98235659775 m
    

     99%|████████████████████████████████████████████████████████████████████████████▉ | 1243/1261 [03:07<00:02,  6.65it/s]

    The radius of curvature is: 4240.06494585088 m 74616.80343462605 m
    

     99%|████████████████████████████████████████████████████████████████████████████▉ | 1244/1261 [03:07<00:02,  6.65it/s]

    The radius of curvature is: 3891.6394737662886 m 13436.459666206347 m
    

     99%|█████████████████████████████████████████████████████████████████████████████ | 1245/1261 [03:07<00:02,  6.65it/s]

    The radius of curvature is: 7265.816342540787 m 1841.987471812776 m
    

     99%|█████████████████████████████████████████████████████████████████████████████ | 1246/1261 [03:07<00:02,  6.65it/s]

    The radius of curvature is: 22096.510258571878 m 2536.556722491478 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▏| 1247/1261 [03:07<00:02,  6.65it/s]

    The radius of curvature is: 6537.933585568527 m 1543.0595354233421 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▏| 1248/1261 [03:07<00:01,  6.65it/s]

    The radius of curvature is: 11715.777658268144 m 1719.773522074514 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▎| 1249/1261 [03:07<00:01,  6.65it/s]

    The radius of curvature is: 38690.38602776534 m 4845.042533685969 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▎| 1250/1261 [03:08<00:01,  6.65it/s]

    The radius of curvature is: 7921.340051873622 m 2811.983034851892 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▍| 1251/1261 [03:08<00:01,  6.65it/s]

    The radius of curvature is: 4668.50854571744 m 17841.919095551406 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▍| 1252/1261 [03:08<00:01,  6.65it/s]

    The radius of curvature is: 5796.911636149868 m 1910.5006525871097 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▌| 1253/1261 [03:08<00:01,  6.65it/s]

    The radius of curvature is: 6763.689847158619 m 8383.729124644651 m
    

     99%|█████████████████████████████████████████████████████████████████████████████▌| 1254/1261 [03:08<00:01,  6.65it/s]

    The radius of curvature is: 2132.969367823326 m 3419.1935491572704 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▋| 1255/1261 [03:08<00:00,  6.65it/s]

    The radius of curvature is: 6550.704757394813 m 2265.600322452135 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▋| 1256/1261 [03:08<00:00,  6.65it/s]

    The radius of curvature is: 7818.534047517991 m 20027.611991713966 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▊| 1257/1261 [03:09<00:00,  6.65it/s]

    The radius of curvature is: 14202.58558919137 m 1811.8873537914612 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▊| 1258/1261 [03:09<00:00,  6.65it/s]

    The radius of curvature is: 14900.78092122335 m 1857.869971840516 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▉| 1259/1261 [03:09<00:00,  6.65it/s]

    The radius of curvature is: 33934.0119379794 m 1279.109057116314 m
    

    100%|█████████████████████████████████████████████████████████████████████████████▉| 1260/1261 [03:09<00:00,  6.65it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: testvideo.mp4 
    
    Wall time: 3min 9s
    


##### Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Challenges:

Color space transformations: I struggled to find the correct combination of color, gradient, and binary thresholds that would allow for yellow and white lane detection in lit and shaded as well as against different pavements. This really highlights the difficulty in creating a universal solution since there was a lot of variation in conditions during just 50 seconds of driving. Additionally I struggled to find the right threshold values for each transform/gradient. A helper app with sliders that allow for quicker experimentation of values would have sped this up significantly. 

Pipeline Weaknesses:

Changing light conditions: The current pipeline is manually tuned for well lit conditions. Driving at night or in bad weather will have negative effects on the ability of this pipeline to identify lane lines.

Tight Radius Turns: Due to extensive use of polyline averaging I don't think this pipeline will work well for tight turns.

Pavement Color: Pavement color can negatively impact the ability of the pipeline to discern lane lines. For example a red(HOV Lane in VA) or green(Bike lanes in DC) paved road could blend in with a yellow lane thus making it invisible to the algorithm.

Faded Lane Lines: Not all lane lines are created equal! Some municipalities struggle to maintain large road networks causing some lane lines to become discernable over time. My pipeline would struggle to identify these worn out lanes.

Obfuscated Lane Lines: If there is an object on the road covering the lane line my pipeline will not be able to infer that there is a lane line under the object.





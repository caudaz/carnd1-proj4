
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle

#Generate images without having a window appear
import matplotlib
matplotlib.use('Agg')

####################  CAMERA CALIBRATION  #####################################

#inside corner count on chessboard
nx = 9
ny =6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

if not os.path.exists('camera_cal_corners_found'):
    os.makedirs('camera_cal_corners_found')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        write_name = 'camera_cal_corners_found/corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/camera_calibration_pickle.p", "wb" ) )


####################  THRESHOLD IMAGE  ########################################

# Edit this function to create your own pipeline.
def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    combined = np.zeros_like(s_channel)
    combined[ (sxbinary == 1) | (s_binary==1) ] = 1
    return combined


####################  TRANSFORM IMAGE  ########################################    

def ptransf(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


####################  PIPELINE TEST1  #########################################

for file in glob.glob('test_images/*'):
    # Test undistortion on a NEW image
    img               = cv2.imread(file)
    undst             = cv2.undistort(img, mtx, dist, None, mtx)
    image_thr         = threshold(undst)
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[595, 450], 
                      [685, 450], 
                      [1110,720], 
                      [205, 720]])    
    offsetX = 320
    offsetY = 0
    dst = np.float32([[offsetX              ,               offsetY],
                      [img_size[0] - offsetX,               offsetY],
                      [img_size[0] - offsetX, img_size[1] - offsetY],
                      [offsetX              , img_size[1] - offsetY]])    
    image_thr_ptransf = ptransf(image_thr, src, dst)
    # Plot the result
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original', fontsize=20)
    ax2.imshow(cv2.cvtColor(undst, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted', fontsize=20)    
    ax3.imshow(image_thr, cmap='gray')
    ax3.set_title('Thresholded', fontsize=20)    
    ax4.imshow(image_thr_ptransf, cmap='gray')
    ax4.set_title('Perspective Transform', fontsize=20)
    plt.show()


####################  FIND LANES BY CONVOLUTION  ##############################

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def show_window_centroids(window_centroids, image, window_width, window_height, margin):    
    # If we found any window centers
    if len(window_centroids) > 0:    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255    
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((image,image,image)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((image,image,image)),np.uint8)    
    # Display the final results
    plt.imshow(output)
    plt.title('Window Fitting Results')
    plt.show()
    
for file in glob.glob('test_images_undistort/*'):
    image             = mpimg.imread(file)
    image_thr         = threshold(image)
    src = np.float32([[595, 450], 
                      [685, 450], 
                      [1110,720], 
                      [205, 720]])    
    offsetX = 320
    offsetY = 0
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32([[offsetX              ,               offsetY],
                      [img_size[0] - offsetX,               offsetY],
                      [img_size[0] - offsetX, img_size[1] - offsetY],
                      [offsetX              , img_size[1] - offsetY]])    
    image_thr_ptransf = ptransf(image_thr, src, dst)
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    image_thr_ptransf_bin = np.zeros_like(image_thr_ptransf).astype(dtype='uint8')
    image_thr_ptransf_bin[image_thr_ptransf>0.0] = 255
    window_centroids = find_window_centroids(image_thr_ptransf_bin, window_width, window_height, margin)
    show_window_centroids(window_centroids, image_thr_ptransf_bin, window_width, window_height, margin)


####################  FIND LANES BY SLIDING WINDOWS  ##########################

def sliding_windows1(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    #VISUALIZE
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]   
    # Black background image with lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title('Sliding Windows 1 Results')
    plt.show()
      
    return out_img, left_fit, right_fit, left_fitx, right_fitx


def sliding_windows2(binary_warped, left_fit, right_fit):
    #binary_warped = mpimg.imread('binary_warped.jpg')
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title('Sliding Windows 2 Results')
    plt.show()

    return result, left_fit, right_fit, left_fitx, right_fitx


def sliding_windows2b(binary_warped, left_fit, right_fit):
    # same as sliding_windows2 but highlights entire lane on image
    #binary_warped = mpimg.imread('binary_warped.jpg')
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
#    # Generate a polygon to illustrate the search window area
#    # And recast the x and y points into usable format for cv2.fillPoly()
#    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
#    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
#    left_line_pts = np.hstack((left_line_window1, left_line_window2))
#    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
#    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
#    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()   
    left_line_window  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    center_line_pts = np.hstack((left_line_window, right_line_window))
    
    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([center_line_pts]), (0,255, 0))    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title('Sliding Windows 2 Results')
    plt.show()

    return result, left_fit, right_fit, left_fitx, right_fitx


for file in glob.glob('test_images_undistort/*'):
    image             = mpimg.imread(file)
    image_thr         = threshold(image)
    src = np.float32([[595, 450], 
                      [685, 450], 
                      [1110,720], 
                      [205, 720]])    
    offsetX = 320
    offsetY = 0
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32([[offsetX              ,               offsetY],
                      [img_size[0] - offsetX,               offsetY],
                      [img_size[0] - offsetX, img_size[1] - offsetY],
                      [offsetX              , img_size[1] - offsetY]])    
    image_thr_ptransf = ptransf(image_thr, src, dst)
    image_thr_ptransf_bin = np.zeros_like(image_thr_ptransf).astype(dtype='uint8')
    image_thr_ptransf_bin[image_thr_ptransf>0.0] = 255
    image_lanes, left_fit, right_fit, left_fitx, right_fitx = sliding_windows1(image_thr_ptransf)
    image_lanes, left_fit, right_fit, left_fitx, right_fitx = sliding_windows2(image_thr_ptransf, left_fit, right_fit)    


###############################################################################

video = 'project_video.mp4'
#video = 'challenge_video.mp4'
#video = 'harder_challenge_video.mp4'

video_name, video_extension = os.path.splitext(video)
video_folder              = 'output_images'
video_folder_pre          = 'output_images/'+video_name + '_pre'
video_folder_post         = 'output_images/'+video_name + '_post'
video_folder_post_ptransf = 'output_images/'+video_name + '_post_ptransf'
video_folder_combined     = 'output_images/'+video_name + '_combined'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)
if not os.path.exists(video_folder_pre):
    os.makedirs(video_folder_pre)
if not os.path.exists(video_folder_post):
    os.makedirs(video_folder_post)
if not os.path.exists(video_folder_post_ptransf):
    os.makedirs(video_folder_post_ptransf)
if not os.path.exists(video_folder_combined):
    os.makedirs(video_folder_combined)
    
# Convert MP4 to JPGs
convert_mp4_to_jpgs = True
if convert_mp4_to_jpgs:
    vidcap = cv2.VideoCapture(video)
    fps    = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
      success, image = vidcap.read()
      if success:
         cv2.imwrite(video_folder_pre+'/frame{0:05}.jpg'.format(count), image)
      count += 1
    vidcap.release()


#######  PIPELINE  ##########

# Car center x-dir in video frame (720p)
car_center_x_pos = 665

# Generate some fake data to represent lane-line pixels
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
ave_curverad_list=[]
car_center_in_m_list=[]

i = 0 
files = os.listdir(video_folder_pre)
for file in files[0:-1]:
    image             = mpimg.imread(video_folder_pre+'/'+file)
    imageRGB          = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undst             = cv2.undistort(imageRGB, mtx, dist, None, mtx)
    image_thr         = threshold(undst, s_thresh=(100, 255), sx_thresh=(20, 100))
    src = np.float32([[595, 450], 
                      [685, 450], 
                      [1110,720], 
                      [205, 720]])    
    offsetX = 320
    offsetY = 0
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32([[offsetX              ,               offsetY],
                      [img_size[0] - offsetX,               offsetY],
                      [img_size[0] - offsetX, img_size[1] - offsetY],
                      [offsetX              , img_size[1] - offsetY]])    
    image_thr_ptransf = ptransf(image_thr, src, dst)
    image_thr_ptransf_bin = np.zeros_like(image_thr_ptransf).astype(dtype='uint8')
    image_thr_ptransf_bin[image_thr_ptransf>0.0] = 255
    print(i, os.path.splitext(os.path.basename(file))[0])
    if i == 0:
        image_lanes, left_fit, right_fit, left_fitx, right_fitx = sliding_windows1(image_thr_ptransf)
        cv2.imwrite(video_folder_post+'/'+os.path.splitext(os.path.basename(file))[0]+'.jpg', image_lanes)        
        i = i + 1
    else:
        image_lanes, left_fit, right_fit, left_fitx, right_fitx = sliding_windows2b(image_thr_ptransf, left_fit, right_fit)    
        cv2.imwrite(video_folder_post+'/'+os.path.splitext(os.path.basename(file))[0]+'.jpg', image_lanes)  
        i = i + 1
    # Fit new polynomials to x,y in world space
    left_fit_cr  = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    ave_curverad = (left_curverad + right_curverad) / 2.0
    ave_curverad_list.append(ave_curverad)                   
    # Car center
    car_center_in_pixels = car_center_x_pos - (left_fitx[-1] + right_fitx[-1]) / 2
    car_center_in_m = car_center_in_pixels * xm_per_pix      
    car_center_in_m_list.append(car_center_in_m)                                   
    # transformed back image
    image_lanes_post_ptransf = ptransf(image_lanes, dst, src).astype('uint8')
    cv2.imwrite(video_folder_post_ptransf+'/'+os.path.splitext(os.path.basename(file))[0]+'.jpg', image_lanes_post_ptransf) 
    # combined image
    image_combined = cv2.addWeighted(undst, 0.6, image_lanes_post_ptransf, 0.4, 0)
    # average values every 5 frames
    if i<=5:
        ave_curverad_str = 'Radius of Curvature = ' + str('{:5d}'.format(int(ave_curverad))) + '[m]'
        cv2.putText(image_combined, ave_curverad_str, (115,75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        if car_center_in_m > 0 :        
            car_center_str = 'Car is ' + '{:1.2f}'.format(abs(car_center_in_m)) + '[m] Right  of the Center'
        else :
            car_center_str = 'Car is ' + '{:1.2f}'.format(abs(car_center_in_m)) + '[m] Left of the Center'
        cv2.putText(image_combined, car_center_str,  (115,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    if i%5 ==0 and i>0 :
        # curvature
        ave_curverad_str_5ave = np.mean(ave_curverad_list[-5:])
        if ave_curverad_str_5ave> 5000:
            ave_curverad_str_5ave = 5000.0
        ave_curverad_str = 'Radius of Curvature = ' + str('{:5d}'.format(int(ave_curverad_str_5ave))) + '[m]'
        # car center
        car_center_str_5ave    = np.mean(car_center_in_m_list[-5:])
        if car_center_in_m > 0 :        
            car_center_str = 'Car is ' + '{:1.2f}'.format(abs(car_center_str_5ave)) + '[m] Right  of the Center'
        else :
            car_center_str = 'Car is ' + '{:1.2f}'.format(abs(car_center_str_5ave)) + '[m] Left of the Center'   
    if i>5:
        cv2.putText(image_combined, ave_curverad_str, (115,75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image_combined, car_center_str,  (115,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # write JPG
    cv2.imwrite(video_folder_combined+'/'+os.path.splitext(os.path.basename(file))[0]+'.jpg', image_combined)   

# Convert JPGs to MP4   
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
format = "XVID"
size = None
is_color=True
fourcc = VideoWriter_fourcc(*format)
vid = None
outvid = os.path.splitext(os.path.basename(video))[0]+'_output.mp4'
for file in glob.glob(video_folder_combined+'/*.jpg'):
    img = imread(file)
    if vid is None:
        if size is None:
            size = img.shape[1], img.shape[0]
        vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    if size[0] != img.shape[1] and size[1] != img.shape[0]:
        img = resize(img, size)
    vid.write(img)
vid.release()
    
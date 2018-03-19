import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


# undistort image
def undistort(img):
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    cal_img = plt.imread("camera_cal/calibration3.jpg")

    gray = cv2.cvtColor(cal_img, cv2.COLOR_RGB2GRAY)

    undistorted = None

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        imgpoints = [corners]
        objpoints = [objp]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted

def color_threshold(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hsl_img[:, :, 2]
    l_channel = hsl_img[:, :, 1]
    # coverting to binary
    sthresh = (60, 255)
    lthresh = (150, 255)
    b_img = np.zeros_like(s_channel)
    b_img[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) & (l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1
    return b_img


def abs_sobel_thresh(gray, orient='x',thresh=(0,255)):
    # Apply the following steps to img
    # 1) Convert to grayscale


    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    ab_sobelx = np.absolute(sobelx)
    ab_sobely = np.absolute(sobely)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_x = np.uint8(255 * ab_sobelx / np.max(ab_sobelx))
    scaled_y = np.uint8(255 * ab_sobely / np.max(ab_sobely))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max

    if orient == 'x':
        sxbinary = np.zeros_like(scaled_x)
        sxbinary[(scaled_x >= thresh[0]) & (scaled_x <= thresh[1])] = 1
        return sxbinary
    else:
        sybinary = np.zeros_like(scaled_y)
        sybinary[(scaled_y >= thresh[0]) & (scaled_y <= thresh[1])] = 1
        return sybinary

def find_mag(gray,sobel_x,sobel_y):
    mag_thresh=(100,170)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = np.uint8((gradmag / scale_factor))

    binary_output = np.zeros_like(gray)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    cv2.imshow("binary",binary_output)
    return binary_output


def find_dir(gray,sobel_x,sobel_y):
    thresh=(0.7,1.2)
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(gray)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def find_gradient(img):
    b_img=color_threshold(img)

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # applying sobel

    gradx = abs_sobel_thresh(gray, orient='x',  thresh=(70, 150))
    grady =abs_sobel_thresh(gray, orient='y',  thresh=(70, 150))
    mag_binary=find_mag(gray,sobel_x,sobel_y)
    dir_binary=find_dir(gray,sobel_x,sobel_y)

    b_sobel_img = np.zeros_like(dir_binary)
    b_sobel_img[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    img_binary = np.zeros_like(dir_binary)
    img_binary[(b_img == 1) | (b_sobel_img == 1)] = 255
    return img_binary


# transform image
def transform(img_binary, trans_type):
    rows,cols=img_binary.shape[:2]
    bottom_left = [cols * 0.20, rows]
    top_left = [cols * 0.47, rows * 0.62]
    bottom_right = [cols * 0.87, rows]
    top_right = [cols * 0.53, rows * 0.62]
    src = np.float32([bottom_left, bottom_right, top_right, top_left])
    dst = np.float32(
        [
            [cols * 0.25, rows],  # bottom_left
            [cols * 0.75, rows],  # bottom_right
            [cols * 0.75, rows * 0],  # top_right
            [cols * 0.25, rows * 0],  # top_left
        ])
    if (trans_type == "forward"):

        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        img_size = (img_binary.shape[1], img_binary.shape[0])
        warped = cv2.warpPerspective(img_binary, M, img_size)

    elif (trans_type == "inverse"):
        M = cv2.getPerspectiveTransform( dst,src)
        # Warp the image using OpenCV warpPerspective()
        img_size = (img_binary.shape[1], img_binary.shape[0])
        warped = cv2.warpPerspective(img_binary, M, img_size)

    return warped


def draw_lanes(binary_warped):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # plt.imshow(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &(nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &(nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)



    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Fit a second order polynomial to each

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = None
    pts_right = None


    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])


    pts = np.hstack((pts_left, pts_right))

    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    full = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # print curvature on the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    c#calculation for car position
    l_fit = left_fit
    r_fit = right_fit
    h = frame.shape[0]
    if right_fit is not None and left_fit is not None:
        car_position = frame.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix


    return full,left_curverad,right_curverad,center_dist


def main(frame):
    # undistory frame
    img = undistort(frame)

    # find gradients
    img_binary = find_gradient(img)


    # transform
    binary_warped = transform(img_binary, "forward")

    # draw lanes
    lane_img,left_curverad,right_curverad,center_dist = draw_lanes(binary_warped)

    lane_img = transform(lane_img, "inverse")


    # draw lane on original image
    complete_img = cv2.addWeighted(frame, 1.0, lane_img, 0.6, 0.0)


    direction=None
    if center_dist<0:
        direction="left"
    else:
        direction="right"

    cv2.putText(complete_img, "Curve Radius :" + str((left_curverad+right_curverad)//2), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 1,
                cv2.LINE_AA)
    cv2.putText(complete_img, "Car is {:4.2f} m {} from center".format(abs(center_dist),direction), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1,
                cv2.LINE_AA)

    return complete_img



if __name__=="__main__":
    cap = cv2.VideoCapture("project_video.mp4")
    writer = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*'XVID'),
                             25, (int(cap.get(3)),int(cap.get(4))))
    while True:
        _, frame = cap.read()
        if (not _):
            break
        out_frame = main(frame)
        writer.write(out_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
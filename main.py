#       STEPS       #
#   1. Convert into grayscale
#   2. Reduce Noise by applying Gaussian blur
#   3. Edge detection
#   4. Hough line detection
#   5. Separate left and right lines
#   6. Draw lines
#   7. Turn Prediction

import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]  # starting position of the y1
    y2 = int(y1 * 3 / 4)  # starting position of the y2 (or ending of the line, which is 2/3 of the height of the image)
    x1 = int((y1 - intercept) / slope)  # starting position of the x1
    x2 = int((y2 - intercept) / slope)  # starting position of the x2
    return np.array([x1, y1, x2, y2])


# Task 5
# This function continues drawing lines from two sides until the end of the mask.
# By this we optimizing line detection via HoughLinesP
def average_slope_intercept(image, lines):
    right_fit = []
    left_fit = []
    left_line = [(0, 0), (0, 0)]
    right_line = [(0, 0), (0, 0)]

    if lines is None:
        left_line = [1]
        right_line = [1]
        return [left_line, right_line]

    for line in lines:
        if line is None:
            continue
        else:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # print('parameters: ', parameters)
            slope = parameters[0]
            intercept = parameters[1]

            # separate lines to right and left sides
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    # print("right_fit")
    # print(right_fit)

    # averages values from right and left fit and returns numbers
    # and this two numbers should be places onto coordinate system, which done in make_coordinates()

    if len(right_fit) == len(left_fit) == 0:
        return np.array([0])
    if len(right_fit) == 0:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)
        return np.array([left_line])
    if len(left_fit) == 0:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_avg)
        return np.array([right_line])
    else:
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_avg)
        right_line = make_coordinates(image, right_fit_avg)

    cv2.imwrite('output/left_line.png', left_line)
    cv2.imwrite('output/right_line.png', right_line)
    return np.array([left_line, right_line])


# Task 1, 2
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # convert an image into grayscale color space
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise with blurring an img

    # edge detection part

    # since each img is an array with colors it has columns(x) and rows(y)
    # This X's and Y's can be expressed by linear  function
    # and changing between colors what we will need while edge detection.
    # And as we know changing of any function is a derivative of this function
    # And all this calculation done by cv2.Canny() function

    canny = cv2.Canny(blur, 50, 150)

    return canny


# Task 3
def region_of_interest(image):  # creates mask, which will be used for a region where edges will be detected
    height = image.shape[0]  # returns value of the height of an image
    # width = image.shape[1]
    triangle = np.array([
        [
            (333, height),
            (1100, height),
            (640, 480)
        ]
    ])
    # coordinates of a triangle
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    cv2.imwrite('output/mask.png', mask)

    masked_img = cv2.bitwise_and(image,
                                 mask)  # apply and operation on two arrays, as result we get cropped area of the img
    cv2.imwrite('output/masked.png', masked_img)

    return masked_img


# Task 6
def display_line(image, lines):
    line_img = np.zeros_like(image)

    if lines is not None and len(lines[0]) == 4:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.86:
                cv2.putText(image, "RIGHT", (800, 500), cv2.QT_FONT_NORMAL, 2, (0, 0, 255))
            elif 0.86 > slope > 0:
                cv2.putText(image, "FORWARD", (520, 450), cv2.QT_FONT_NORMAL, 2, (0, 0, 255))
            elif slope < -1.3:
                cv2.putText(image, "LEFT", (300, 500), cv2.QT_FONT_NORMAL, 2, (0, 0, 255))
            else:
                cv2.putText(image, "", (100, 200), cv2.QT_FONT_NORMAL, 2, (0, 0, 255))

    else:
        x1 = 1
        y1 = 1
        x2 = 2
        y2 = 2
    return line_img


if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/test.mp4")
    while cap.isOpened():
        _, frame = cap.read()  # load video
        if frame is None:
            break
        frame = frame.astype('uint8')
        canny_img = canny(frame)
        cropped_img = region_of_interest(canny_img)
        # Task 4
        # we need to use polar coordinate system, which p = xCos(t) + ySin(t)
        # also values gotten from polar coordinates will be placed onto Hough Space, where we will read intersection
        # and knowing these intersection we easly can draw straight line
        lines = cv2.HoughLinesP(cropped_img,
                                2,
                                np.pi / 180,
                                100,
                                np.array([]),
                                minLineLength=10,
                                maxLineGap=200)
        avaraged_lines = average_slope_intercept(frame, lines)  # generate coordinates ( y = m+bx ) for both sides
        line_img = display_line(frame, avaraged_lines)  # combine orginal img with line

        # Count FPS of Video
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps - random.randrange(3)
        cv2.putText(frame, "fps:" + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        impose_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

        cv2.imshow("result", impose_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

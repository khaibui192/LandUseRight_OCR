from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np


class processImg:
    
    """
    This class is for cropping the table method:
    - It dilates the image and finds the contours
    - Find the maximum of the contours which is a rectangle and  
    """

    def __init__(self, img_path):
        self.image = cv2.imread(img_path)
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=1) # chỉnh iteration

    # def convert_image_to_grayscale(self): # chuyển xám
    #     self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     return self.grayscale_image

    # def threshold_image(self): #giữ pixel trắng và xám
    #     self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     return self.thresholded_image

    # def invert_image(self):
    #     self.inverted_image = cv2.bitwise_not(self.thresholded_image)
    #     return self.inverted_image

    # def dilate_image(self):
    #     self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=1) # chỉnh iteration
    #     return self.dilated_image

    def order_points(self, pts):
        # pts is the max contour of an image. This is for a table so it's supposedly rectangle
        pts = pts.reshape(4, 2) # reshape to 4x2
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1) # sum up all the the array horizontally, returns a matrix with 1 column which is the left side of rectangle.
        # the min value of that 1 column matrix represent the sum of 2 top corner and every point between them
        # the max value of that 1 column matrix represent the sum of 2 bottom corner and every point between them
        
        rect[0] = pts[np.argmin(s)] # gets the indices of min of s which means the top left coordinate
        rect[2] = pts[np.argmax(s)] # gets the indices of max of s which means the bottom left coordinate
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def find_contours(self):
        contours, hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # gray = process_img.convert_image_to_grayscale()
        # image_with_all_contours = gray.copy()
        # cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
        return contours, hierarchy

    def rec_contour(self):
        contours, hierarchy = self.find_contours()
        rectangular_contours = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rectangular_contours.append(approx)
        # image_with_only_rectangular_contours = self.grayscale_image.copy()
        # rec_cont = cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
        return rectangular_contours

    def find_MaxContour(self):
        max_area = 0
        contour_with_max_area = None
        rectangular_contours = self.rec_contour()
        for contour in rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                contour_with_max_area = contour
        # image_with_contour_with_max_area = gray.copy()
        # cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)
        return contour_with_max_area

    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def angle_points(self):
        contour_with_max_area = self.find_MaxContour()
        contour_with_max_area_ordered = self.order_points(contour_with_max_area)
        # image_with_points_plotted = self.grayscale_image.copy()
        # for point in contour_with_max_area_ordered:
        #     point_coordinates = (int(point[0]), int(point[1]))
        #     image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
        return contour_with_max_area_ordered

    def calculate_new_image_size(self):
        existing_image_width = self.grayscale_image.shape[1]
        contour_with_max_area_ordered = self.angle_points()
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[3])
        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
        new_image_width = existing_image_width_reduced_by_10_percent
        new_image_height = int(new_image_width * aspect_ratio)
        return new_image_width, new_image_height

    def new_padded_perspective(self):
        contour_with_max_area_ordered = self.angle_points()
        new_image_width, new_image_height = self.calculate_new_image_size()
        pts1 = np.float32(contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_corrected_image = cv2.warpPerspective(self.grayscale_image, matrix, (new_image_width, new_image_height))
        image_height = self.grayscale_image.shape[0]
        padding = int(image_height * 0.1)
        perspective_corrected_image_padded = cv2.copyMakeBorder(perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return perspective_corrected_image_padded
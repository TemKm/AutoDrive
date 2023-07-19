import cv2
import numpy as np
import math
import logging

class JdOpencvLaneDetect(object):
    def __init__(self):
        self.curr_steering_angle = 90

    def detect_edges(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([40, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        edges = cv2.Canny(mask, 200, 400)
        return edges

    def region_of_interest(self, canny):
        height, width = canny.shape
        mask = np.zeros_like(canny)
        polygon = np.array([[
            (0, height*(1/2)),
            (width, height*(1/2)),
            (width, height),
            (0, height),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image

    def detect_line_segments(self, cropped_edges):
        rho = 1
        angle = np.pi / 180
        min_threshold = 10
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=15, maxLineGap=4)
        return line_segments

    def average_slope_intercept(self, frame, line_segments):
        lane_lines = []
        if line_segments is None:
            logging.info('No line_segment segments detected')
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1/3
        left_region_boundary = width * (1 - boundary)
        right_region_boundary = width * boundary

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        if slope < -0.75:
                            left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        if slope > 0.75:
                            right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        return lane_lines

    def make_points(self, frame, line):
        slope, intercept = line
        y1 = int(frame.shape[0])# bottom of the frame
        y2 = int(y1*3/5)         # slightly lower than the middle
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return [[x1, y1, x2, y2]]

    # insert the provided functions here

    def process(self, frame):
        edges = self.detect_edges(frame)
        cropped_edges = self.region_of_interest(edges)
        line_segments = self.detect_line_segments(cropped_edges)
        lane_lines = self.average_slope_intercept(frame, line_segments)
        steering_angle = self.compute_steering_angle(frame, lane_lines)
        final_frame = self.display_lines(frame, lane_lines)
        return steering_angle, final_frame

if __name__ == "__main__":
    lane_detector = JdOpencvLaneDetect()
    frame = cv2.imread("lane.jpg")
    steering_angle, final_frame = lane_detector.process(frame)
    print(f"Steering Angle: {steering_angle}")
    cv2.imshow("Lane Lines", final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

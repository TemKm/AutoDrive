# Modified Python code
import cv2
import numpy as np

def resize_frame(frame, height=500):
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    width = int(height * aspect_ratio)
    return cv2.resize(frame, (width, height))

def process_frame(frame):
    lower_white = np.array([150, 150, 130])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(frame, lower_white, upper_white)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    return result, edges

def roi(edges):
    mask = np.zeros_like(edges)
    h, w = edges.shape
    roi_vertices = np.array([[(w // 3, h), (w // 3, h // 4 * 3), (w // 3 * 2, h // 4 * 3), (w // 3 * 2, h)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)

def hough_lines(edges):
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)

def select_representative_line(group):
    longest_line = max(group, key=lambda line: np.sqrt((line[0][0] - line[0][2]) ** 2 + (line[0][1] - line[0][3]) ** 2))
    return longest_line

def group_lines(lines, w, angle_threshold=15, distance_threshold=20):
    grouped_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < w // 2 and x2 < w // 2:  # Left side
            side = 0
        elif x1 > w // 2 and x2 > w // 2:  # Right side
            side = 1
        else:
            continue

        matched_group = None
        for group in grouped_lines:
            if group["side"] != side:
                continue
            for group_line in group["lines"]:
                gx1, gy1, gx2, gy2 = group_line[0]
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angle2 = np.arctan2(gy2 - gy1, gx2 - gx1) * 180 / np.pi

                if abs(angle1 - angle2) < angle_threshold:
                    dist1 = np.sqrt((x1 - gx1) ** 2 + (y1 - gy1) ** 2)
                    dist2 = np.sqrt((x2 - gx2) ** 2 + (y2 - gy2) ** 2)
                    if dist1 < distance_threshold or dist2 < distance_threshold:
                        matched_group = group
                        break

        if matched_group is not None:
            matched_group["lines"].append(line)
        else:
            grouped_lines.append({"side": side, "lines": [line]})

    representative_lines = []
    for group in grouped_lines:
        representative_lines.append(select_representative_line(group["lines"]))

    return representative_lines

def draw_lines(frame, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame

cap = cv2.VideoCapture("drive3.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result, edges = process_frame(frame)
    resized_result = resize_frame(result)
    resized_edges = resize_frame(edges)
    cv2.imshow("Result", resized_result)
    cv2.imshow("Edges", resized_edges)

    roi_edges = roi(edges)
    resized_roi_edges = resize_frame(roi_edges)
    cv2.imshow("ROI Edges", resized_roi_edges)

    lines = hough_lines(roi_edges)

    if lines is not None:
        representative_lines = group_lines(lines, frame.shape[1])
    else:
        representative_lines = []

    lines_img = draw_lines(frame.copy(), representative_lines)
    resized_lines_img = resize_frame(lines_img)
    cv2.imshow("Hough Lines", resized_lines_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
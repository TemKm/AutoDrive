import cv2
import numpy as np

def resize_frame(frame, height=500):
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    width = int(height * aspect_ratio)
    return cv2.resize(frame, (width, height))

def process_frame(frame):
    # BGR 영상에서 흰색 후보 검출
    lower_white = np.array([130, 130, 130])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(frame, lower_white, upper_white)

    # HSV 영상에서 노란색 차선 후보 검출
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 검출한 후보 합치기
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 필터링 및 마스킹
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # 영상을 GrayScale로 변환
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection으로 엣지 추출
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

def group_lines(lines, angle_threshold=15, distance_threshold=20):
    grouped_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        matched_group = None
        for group in grouped_lines:
            for group_line in group:
                gx1, gy1, gx2, gy2 = group_line[0]
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angle2 = np.arctan2(gy2 - gy1, gx2 - gx1) * 180 / np.pi

                if abs(angle1 - angle2) < angle_threshold:
                    dist1 = np.sqrt((x1 - gx1)**2 + (y1 - gy1)**2)
                    dist2 = np.sqrt((x2 - gx2)**2 + (y2 - gy2)**2)
                    if dist1 < distance_threshold or dist2 < distance_threshold:
                        matched_group = group
                        break

        if matched_group is not None:
            matched_group.append(line)
        else:
            grouped_lines.append([line])

    return grouped_lines

def draw_lines(frame, grouped_lines):
    for group in grouped_lines:
        if group is not None:
            points = []
            for line in group:
                x1, y1, x2, y2 = line[0]
                points.extend([(x1, y1), (x2, y2)])

            points = np.array(points, dtype=np.int32)
            curve = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = curve[:, 0]
            h, w = frame.shape[:2]
            x1, y1 = int(x0 - w * vx), int(y0 - w * vy)
            x2, y2 = int(x0 + w * vx), int(y0 + w * vy)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame

# 웹캠에서 영상 가져오기
cap = cv2.VideoCapture("drive2.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 프레임 처리 및 결과 창에 띄우기
    result, edges = process_frame(frame)
    resized_result = resize_frame(result)
    resized_edges = resize_frame(edges)
    cv2.imshow("Result", resized_result)
    cv2.imshow("Edges", resized_edges)

    # 관심 영역을 적용한 엣지 추출
    roi_edges = roi(edges)
    resized_roi_edges = resize_frame(roi_edges)
    cv2.imshow("ROI Edges", resized_roi_edges)

    # 허프 변환을 사용하여 직선 성분 추출
    lines = hough_lines(roi_edges)

    if lines is not None:
        # 추출한 직선 성분을 그룹화
        grouped_lines = group_lines(lines)
    else:
        grouped_lines = []

    # 추출한 직선 성분을 표시
    lines_img = draw_lines(frame.copy(), grouped_lines)
    resized_lines_img = resize_frame(lines_img)
    cv2.imshow("Hough Lines", resized_lines_img)

    # 종료 키 입력 대기
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

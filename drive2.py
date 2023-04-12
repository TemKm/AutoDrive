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
    roi_vertices = np.array([[(w // 3, h), (w // 3, h // 3 * 2), (w // 3 * 2, h // 3 * 2), (w // 3 * 2, h)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)

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

    # 종료 키 입력 대기
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

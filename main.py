import cv2
import numpy as np

def filter_white_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 흰색 범위 설정 및 마스크 생성
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 노란색 범위 설정 및 마스크 생성
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 흰색 및 노란색 마스크 합성
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    return result

# 이미지 파일 읽기
image = cv2.imread('test.jpg')

# 흰색 및 노란색 필터링
filtered_image = filter_white_yellow(image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)

# 그레이스케일 변환
gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

# Canny Edge Detection으로 엣지 추출
canny_image = cv2.Canny(gray_image, 50, 150)
cv2.imshow('Canny Edge Detection', canny_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
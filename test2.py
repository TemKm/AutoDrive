import cv2
import numpy as np

def filter_white_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 흰색 범위 설정 및 마스크 생성
    lower_white = np.array([100, 100, 100], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 노란색 범위 설정 및 마스크 생성
    lower_yellow = np.array([0, 0, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 흰색 및 노란색 마스크 합성
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    return result

# 웹캠에서 영상 가져오기
cap = cv2.VideoCapture('d.mkv')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # 흰색 및 노란색 필터링
        filtered_image = filter_white_yellow(frame)

        # 그레이스케일 변환
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

        # Canny Edge Detection으로 엣지 추출
        canny_image = cv2.Canny(gray_image, 50, 150)
        cv2.imshow('Processed Result', canny_image)

        # 키보드 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

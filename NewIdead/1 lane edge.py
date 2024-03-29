from PIL import ImageGrab
import cv2
import numpy as np

class ScreenCap:
    def __init__(self):
        self.screen_coords = (0, 100, 1280, 720)  # 전체 화면 좌표로 설정. 환경에 맞게 조정 가능.

    def process_frame(self, frame):
        # BGR 영상에서 흰색 후보 검출
        lower_white = np.array([155, 130, 130])
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

        return edges

    def screen_cap(self):
        """
        화면 캡처 및 출력
        """
        print("화면 캡처 및 출력 시작.")
        while True:
            # 화면을 캡처합니다.
            img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=self.screen_coords)), cv2.COLOR_BGR2RGB)
            
            # 캡처한 영상을 처리합니다.
            edges = self.process_frame(img)

            # 처리된 외곽선을 출력합니다.
            cv2.imshow("Edges Capture", edges)

            # 'q' 키를 누르면 루프에서 벗어납니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("화면 캡처 및 출력 중단.")
                cv2.destroyAllWindows()
                break

cap = ScreenCap()
cap.screen_cap()
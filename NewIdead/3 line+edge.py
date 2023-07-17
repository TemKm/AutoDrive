from PIL import ImageGrab
import cv2
import numpy as np

class ScreenCap:
    def __init__(self):
        self.screen_coords = (0, 100, 1280, 720)  # 전체 화면 좌표로 설정. 환경에 맞게 조정 가능.

    def process_frame(self, frame):
        # BGR 영상에서 흰색 후보 검출
        lower_white = np.array([155, 135, 130])
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

            # 채널 분리
            b, g, r = cv2.split(img)
            # edge가 감지된 위치에 빨간색 할당
            r[edges > 0] = 255
            b[edges > 0] = 0
            g[edges > 0] = 0

            # 채널 병합
            img = cv2.merge((b, g, r))

            # 화면 중간의 x 좌표를 계산합니다.
            middle_x = img.shape[1] // 2

            # 이미지에 형광 초록색 선을 그립니다.
            cv2.line(img, (middle_x - 200, img.shape[0] // 2 + 100), (middle_x - 200 + 150, img.shape[0] // 2 + 100), (0, 255, 0), 2)
            cv2.line(img, (middle_x + 100, img.shape[0] // 2 + 100), (middle_x + 100 + 150, img.shape[0] // 2 + 100), (0, 255, 0), 2)

            # 선을 그린 이미지를 출력합니다.
            cv2.imshow("Screen Capture", img)

            # 'q' 키를 누르면 루프에서 벗어납니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("화면 캡처 및 출력 중단.")
                cv2.destroyAllWindows()
                break

cap = ScreenCap()
cap.screen_cap()
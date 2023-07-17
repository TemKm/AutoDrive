from PIL import ImageGrab
import cv2
import numpy as np
from pynput.keyboard import Controller, Key
import threading
import time

class ScreenCap:
    def __init__(self):
        self.screen_coords = (0, 100, 1280, 720)  # 전체 화면 좌표로 설정. 환경에 맞게 조정 가능.
        self.keyboard = Controller()

    def process_frame(self, frame):
        # BGR 영상에서 흰색 후보 검출
        lower_white = np.array([180, 150, 150])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(frame, lower_white, upper_white)

        # 검출한 후보 합치기
        combined_mask = white_mask

        # 필터링 및 마스킹
        result = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # 영상을 GrayScale로 변환
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Canny Edge Detection으로 엣지 추출
        edges = cv2.Canny(gray, 50, 150)

        return edges

    def press_key(self, key, duration):
        """
        키를 주어진 시간 동안 누릅니다. 이 함수는 별도의 스레드에서 실행됩니다.
        """
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)

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

            middle_x = img.shape[1] // 2
            line_y = img.shape[0] // 2 + 100  # 선의 y 좌표

            # 이미지에 형광 초록색 선을 그립니다.
            left_line_start, left_line_end = (middle_x - 300, line_y), (middle_x - 300 + 300, line_y)
            cv2.line(img, left_line_start, left_line_end, (0, 255, 0), 2)

            # 왼쪽 초록색 선을 삼등분합니다.
            left_line_thirds = [left_line_start[0] + (left_line_end[0] - left_line_start[0]) * i // 3 for i in range(1, 3)]

            # 각 삼등분한 지점에서 외곽선이 닿는지 파악합니다.
            touch_points = []
            for i in range(3):
                start = left_line_thirds[i - 1] if i > 0 else left_line_start[0]
                end = left_line_thirds[i] if i < 2 else left_line_end[0]
                for x in range(start, end):
                    if edges[line_y, x] > 0:  # 외곽선이 선과 교차
                        touch_points.append(i)
                        break

            # 외곽선이 닿은 위치에 따라 키를 누르고 화면에 출력합니다.
            
            for i in touch_points:
                if i == 0:
                    direction = "Left"
                    threading.Thread(target=self.press_key, args=('a', 0.0001)).start()
                elif i == 1:
                    direction = "Middle"
                    # 필요한 키 입력 코드를 추가하세요.
                else:
                    direction = "Right"
                    threading.Thread(target=self.press_key, args=('d', 0.0001)).start()
                cv2.putText(img, direction, (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            # 선을 그린 이미지를 출력합니다.
            cv2.imshow("Screen Capture", img)

            # 'q' 키를 누르면 루프에서 벗어납니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("화면 캡처 및 출력 중단.")
                cv2.destroyAllWindows()
                break

cap = ScreenCap()
cap.screen_cap()




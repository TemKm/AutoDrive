from PIL import ImageGrab
import cv2
import numpy as np

class ScreenCap:
    def __init__(self):
        self.screen_coords = [0, 0, 2560, 1440]  # 전체 화면 좌표를 1440p 해상도로 설정
        cv2.namedWindow('Screen Capture')
        cv2.createTrackbar('X1', 'Screen Capture', 0, 2560, self.update_coords)
        cv2.createTrackbar('Y1', 'Screen Capture', 0, 1440, self.update_coords)
        cv2.createTrackbar('X2', 'Screen Capture', 2560, 2560, self.update_coords)
        cv2.createTrackbar('Y2', 'Screen Capture', 1440, 1440, self.update_coords)

    def update_coords(self, val):
        self.screen_coords = [cv2.getTrackbarPos('X1', 'Screen Capture'), 
                              cv2.getTrackbarPos('Y1', 'Screen Capture'), 
                              cv2.getTrackbarPos('X2', 'Screen Capture'), 
                              cv2.getTrackbarPos('Y2', 'Screen Capture')]

    def screen_cap(self):
        """
        화면 캡첐 및 출력
        """
        print("화면 캡처 및 출력 시작.")
        while True:
            # 화면을 캡첐합니다.
            img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=tuple(self.screen_coords))), cv2.COLOR_BGR2RGB)
            # 화면에 캡처한 이미지를 출력합니다.
            cv2.imshow("Screen Capture", img)

            # 'q' 키를 누르면 루프에서 벗어납니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("화면 캡처 및 출력 중단.")
                cv2.destroyAllWindows()
                break

cap = ScreenCap()
cap.screen_cap()

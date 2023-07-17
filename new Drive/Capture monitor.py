from PIL import ImageGrab
import cv2
import numpy as np

class ScreenCap:
    def __init__(self):
        self.screen_coords = (0, 100, 1280, 720)  # 전체 화면 좌표로 설정. 환경에 맞게 조정 가능.

    def screen_cap(self):
        """
        화면 캡쳐 및 출력
        """
        print("화면 캡처 및 출력 시작.")
        while True:
            # 화면을 캡처합니다.
            img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=self.screen_coords)), cv2.COLOR_BGR2RGB)
            # 화면에 캡처한 이미지를 출력합니다.
            cv2.imshow("Screen Capture", img)

            # 'q' 키를 누르면 루프에서 벗어납니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("화면 캡처 및 출력 중단.")
                cv2.destroyAllWindows()
                break

cap = ScreenCap()
cap.screen_cap()
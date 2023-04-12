import cv2
import numpy as np

def create_circular_spectrum(lower_bgr, upper_bgr):
    lower_hsv = cv2.cvtColor(np.uint8([[lower_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    upper_hsv = cv2.cvtColor(np.uint8([[upper_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]

    radius = 300
    width = 600
    height = 600
    h_range = np.linspace(lower_hsv[0], upper_hsv[0], width, dtype=np.uint8)
    s_range = np.linspace(lower_hsv[1], upper_hsv[1], width, dtype=np.uint8)

    hsv_spectrum = np.full((height, width, 3), upper_hsv[2], dtype=np.uint8)
    hsv_spectrum[:, :, 0] = np.tile(h_range, (height, 1))
    hsv_spectrum[:, :, 1] = np.tile(s_range, (height, 1))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)

    hsv_circular_spectrum = cv2.bitwise_and(hsv_spectrum, hsv_spectrum, mask=mask)
    bgr_circular_spectrum = cv2.cvtColor(hsv_circular_spectrum, cv2.COLOR_HSV2BGR)

    return bgr_circular_spectrum

# 첫 번째 BGR 범위
lower_bgr1 = np.array([100, 100, 100], dtype=np.uint8)
upper_bgr1 = np.array([255, 255, 255], dtype=np.uint8)
spectrum1 = create_circular_spectrum(lower_bgr1, upper_bgr1)
cv2.imshow('Circular Spectrum 1', spectrum1)

# 두 번째 BGR 범위 (HSV [20, 100, 100]부터 [30, 255, 255])
lower_bgr2 = cv2.cvtColor(np.uint8([[[20, 100, 100]]]), cv2.COLOR_HSV2BGR)[0, 0]
upper_bgr2 = cv2.cvtColor(np.uint8([[[30, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
spectrum2 = create_circular_spectrum(lower_bgr2, upper_bgr2)
cv2.imshow('Circular Spectrum 2', spectrum2)

cv2.waitKey(0)
cv2.destroyAllWindows()

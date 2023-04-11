import keyboard
import pyautogui

# 마우스 이동 거리 설정
move_distance = 50

def move_left(event):
    # 현재 마우스 위치를 얻습니다.
    x, y = pyautogui.position()
    # 왼쪽으로 이동
    pyautogui.moveTo(x - move_distance, y)

def move_right(event):
    # 현재 마우스 위치를 얻습니다.
    x, y = pyautogui.position()
    # 오른쪽으로 이동
    pyautogui.moveTo(x + move_distance, y)

keyboard.on_press_key('a', move_left)
keyboard.on_press_key('d', move_right)

# 이벤트 리스너를 유지합니다.
keyboard.wait()

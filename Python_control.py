import serial
import time

# 'COM5' 부분에 환경에 맞는 포트 입력
ser = serial.Serial('COM5', 9600)

while True:
    if ser.readable():
        
        # 1 또는 0 입력
        val = input()
        
        # 1을 프롬프트에 입력시 연속 360도 회전
        if val == '1':
            val = val.encode('utf-8')
            ser.write(val)
            print("360' turn")
            
        # 2을 프롬프트에 입력시 연속 -360도 회전
        elif val == '0':
            val = val.encode('utf-8')
            ser.write(val)
            print("-360' turn")
            
import cv2
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet("C:\Pro\AutoDrive\Gpt\yolov3.weights", "C:\Pro\AutoDrive\Gpt\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 클래스 라벨 로드
with open("C:\Pro\AutoDrive\Gpt\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 비디오 파일 로드
cap = cv2.VideoCapture('C:\Pro\AutoDrive\Gpt\Acc.mp4')

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 사각형 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                if w*h > 50000:
                    cv2.putText(img, 'Brake!!', (10, 50), font, 3, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 30), font, 3, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

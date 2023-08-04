import cv2
import time
import numpy as np

colors = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

class_names = []
with open('coco_names.txt','r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("teste_carros.mp4") # aqui vai algum video ou webcam

# carregando os pesos da rede neural
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg') # versao tiny
# net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# criando o modelo
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(416,416), scale=(1/255)) # tiny
# model.setInputParams(size=(608,608), scale=(1/255)) # para o yolo padrao

# Lendo os frames do video
while True:

    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame,0.1,0.2)
    end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = colors[int(classid) % len(colors)]
        label = f"{class_names[classid]}: {score}"
        
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
    
    fps_label = f"FPS: {round((1/(end - start)),2)}"

    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
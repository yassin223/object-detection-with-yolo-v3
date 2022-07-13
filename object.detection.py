import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0) # (0) for your webcam - (1) if you have a second webcam - ("video.... .mp4") - if you have a video

while True:
    frame, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_names = net.getUnconnectedOutLayersNames()
    layers_output = net.forward(output_names)

    boxes = []
    confidences = []
    class_ids = []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(len(boxes))
    id_number = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(id_number)
    font = cv2.FONT_HERSHEY_PLAIN
    couleur = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in id_number:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = couleur[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y), font, 2, (255, 255, 255), 2)
        print(i)

    # for x in blob:
    # for n, img_blob in enumerate(x):
    # cv2.imshow(str(n), img_blob)
    cv2.imshow("image", img)
    #cv2.waitKey(1)
    key = cv2.waitKey(1) 
    if key == 27 or key ==113:
        break
cap.release()
cv2.destroyAllWindows()







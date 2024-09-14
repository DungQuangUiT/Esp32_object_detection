import cv2
import urllib.request
import numpy as np
import concurrent.futures
from ultralytics import YOLOv10
import os


model = YOLOv10(f"{os.getcwd()}/yolov10n.pt")

url='http://192.168.0.122/cam-mid.jpg'
im=None

#cap = cv2.VideoCapture(0)

def run1():
    cv2.namedWindow("esp32 detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)


        # detection
        results = model(im, conf=0.8)
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                c = box.cls
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[int(c)]
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    im,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow('esp32 detection',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()



def run2():
    while cap.isOpened():
        ret, frame = cap.read()
        #recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #set flag to true
        image.flags.writeable = True
        #recolor image back to BGR for rendering (opencv love BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # detection
        results = model(image, conf=0.7)
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                c = box.cls
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[int(c)]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(run1)
            #f2= executer.submit(run2)
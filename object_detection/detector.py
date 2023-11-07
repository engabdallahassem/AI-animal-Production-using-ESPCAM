import numpy as np
import cv2 as cv
import time

CONFICENCE_LEVEL = 0.5 


class Detector:

    def __init__(self):
        # Load names of classes and get random colors
        self.classes = open('yoloy_files/coco.names').read().strip().split('\n')
        print("load {} class".format(len(self.classes)))
        # make random color for every class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        # Give the configuration and weight files for the model and load the network.
        self.net = cv.dnn.readNetFromDarknet('yoloy_files/yolov3.cfg', 'yoloy_files/yolov3.weights')
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def showImage(self,img):
        window_name = 'image'
        cv.imshow(window_name, img)
        cv.waitKey(0)


    def detect(self,img0):
        img = img0.copy()

        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        startTime = time.time()
        outputs = self.net.forward(self.ln)
        takenTime = time.time() - startTime
        print("detect image in {} sec".format(takenTime))

        # combine the 3 output groups into 1 (10647, 85)
        outputs = np.vstack(outputs)
    
        data = self.post_process(img, outputs, CONFICENCE_LEVEL)
        return img,data

    def post_process(self,img, outputs, conf):
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        classIDs = []
        data = {} 

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                _ = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                className = self.classes[classIDs[i]]
                if not className in data:
                    data[className] = 1
                else:
                    data[className] +=1

                text = "{}: {:.1f}%".format(className, confidences[i]*100)
            
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return data


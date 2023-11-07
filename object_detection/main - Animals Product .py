from detector import * 
import cv2 as cv
from gtts import gTTS
import playsound
from threading import Thread
import os
import requests

apiToken = '5838945391:AAFzJfd7geYPSueSpHhrt_u24hcOQUURTy0'
chatID = '2021637139'
apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

def send_photo(frame):
    file_path = "image.jpg"
    cv.imwrite(file_path , frame)
    file_opened = open(file_path, 'rb')
    method = "sendPhoto"
    params = {'chat_id': chatID}
    files = {'photo': file_opened}
    url = f"https://api.telegram.org/bot{apiToken}/sendPhoto?chat_id={chatID}"
    resp = requests.post(url + method, params, files=files)
    return resp
def send_to_telegram(message):



    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

ended = True 
def imageOperation(frame):
    _,data = detctor.detect(frame)
    print(f"image have items {data}")
    for e in data:
        if e in ['bird' , 'cat' , 'dog' , 'horse' , 'sheep' , 'cow', 'elephant' , 'bear' , 'zebra', 'giraffe'] :
            send_to_telegram(f"Animal detected {e}")
            send_photo(_)
    global ended 
    ended = True 

detctor = Detector()
vid = cv.VideoCapture("http://192.168.8.131:81/stream")

while(True):
    ret, frame = vid.read()
    cv.imshow('frame', frame)
    if ended :
        ended = False 
        thread = Thread(target = imageOperation, args = (frame, ))
        thread.start()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
  


from detector import * 
import cv2 as cv
from gtts import gTTS
import os

language = 'en'

  
def textTospeech(text) :
    myobj = gTTS(text=text, lang=language, slow=False)
    print(text)
    myobj.save("text.mp3")
    os.system("text.mp3")


# imageLists = ['cat-dog','fruits',"street","pizza","room"]
imageLists = ["street"]

detctor = Detector(0)
for imgName in imageLists :
    img = cv.imread("images/{}.jpg".format(imgName))
    img,data = detctor.detect(img)
    textTospeech(f"image {imgName} have items {data}")
    # cv.imshow(imgName , img)
 
# cv.waitKey()



  


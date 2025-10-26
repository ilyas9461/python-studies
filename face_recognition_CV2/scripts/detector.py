import cv2
import numpy as np
#import sqlite3
import os
import datetime
import locale
import time
from gtts import gTTS
from pygame import mixer  # Load the popular external library

try:
    locale.setlocale(locale.LC_ALL, '')
except Exception:
    pass

mixer.init()

def speak_tts(audioString):
    print(audioString)
    # tts = gTTS(text=audioString, lang='en')
    # tts.save("audio.mp3")
    mixer.music.load('audio.mp3')
    mixer.music.play()
    time.sleep(3.0)
    mixer.music.stop()
    mixer.music.load('audio_sil.mp3')
    #os.remove("audio.mp3")

def get_uid_names(face_images):
    prev_person_name = ""
    person_name = ""
    id = 0
    labels = []
    for root, dirs, files in os.walk(face_images):
        for file in files:  # check every directory in it
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                person_name = file.split("-")[1]  # os.path.basename(root)
                print(path, person_name)
                if prev_person_name != person_name:  # Check if the name of person has changed
                    labels.append(person_name)
                    id = id+1  # If yes increment the ID count
                    prev_person_name = person_name
    return id, labels

Face_Images = os.path.join(os.getcwd(), "dataset")

#conn = sqlite3.connect('database.db')
#c = conn.cursor()
path_files=os.getcwd()
print(path_files)
fname =path_files+ "/recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)
print(fname)

id, names = get_uid_names(Face_Images)
print(id, names)

from . import get_cascade_file
face_cascade = cv2.CascadeClassifier(get_cascade_file())
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
an = datetime.datetime.now()
format = "%Y-%m-%d %H:%M:%S"

while True:
    timeCheck = time.time()
    ret, img = cap.read()
    matched_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces)>0:

        for (x, y, w, h) in faces:
            cv2.rectangle(matched_img, (x, y), (x+w, y+h), (0,255,0), 3)
            ids, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print(ids, conf)
            name = names[ids]
            if conf >= 50:
                timestamp = datetime.datetime.strftime(datetime.datetime.now(), format)

                cv2.putText(img, timestamp, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150,255,0), 1)
                cv2.putText(img, name, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0), 2)
                try:
                    cv2.imwrite(path_files + '/recognized/' + name + '.jpg', img)
                except Exception:
                    pass

                cv2.putText(matched_img, timestamp, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150,255,0), 1)
                cv2.putText(matched_img, name, (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0), 2)
                cv2.imshow('Face Recognizer', matched_img)
                img = matched_img
            else:
                cv2.putText(img, 'No Match', (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Face Recognizer',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    print('fps - ', 1/(time.time() - timeCheck))

cap.release()
cv2.destroyAllWindows()



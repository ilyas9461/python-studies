import cv2
import numpy as np
import sqlite3
import os
import time

#conn = sqlite3.connect('database.db')

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

#c = conn.cursor()

try:
    # When run as a package module (python -m scripts.record_face)
    from . import get_cascade_file
except Exception:
    # Fallback when running the file directly (python scripts\record_face.py)
    # or when relative imports are not available
    from scripts import get_cascade_file

face_cascade = cv2.CascadeClassifier(get_cascade_file())

# path='./dataset/'+uname+'/'
# if not os.path.exists(path):
#     os.makedirs(path)

#c.execute('INSERT INTO users (name) VALUES (?)', (uname,))

# Tell the program where we have saved the face images
Face_Images = os.path.join(os.getcwd(), "dataset")
print(Face_Images)

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


Face_ID, names = get_uid_names(Face_Images)
print("ID:" + str(Face_ID))
for name in names:
    print("name:" + name)

uid = Face_ID  # c.lastrowid

uname = input("Enter your name: ")
path = './dataset/'

cap = cv2.VideoCapture(0)
time.sleep(2.0)

sampleNum = 0
saving = True

while True:
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    if len(faces)==1 and saving:
            for (x, y, w, h) in faces:
                sampleNum = sampleNum+1
                print(sampleNum)
                cv2.imwrite(path+str(uid)+"-" + str(uname)+"-" +
                            str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                #cv2.waitKey(100)

    cv2.imshow("image", img)

    if sampleNum >= 20:  # 20 is enough
        saving = False
        cv2.putText(img, "Recording finished. Press any key...", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150,255,0),1)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        break
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cap.release()

# conn.commit()
# conn.close()

cv2.destroyAllWindows()

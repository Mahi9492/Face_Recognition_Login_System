import tkinter as tk
from tkinter import *
from tkinter import messagebox as ms
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

from flask import Flask
app = Flask(__name__)


df = pd.read_csv('Profile.csv')
df.sort_values('Ids', inplace = True)
df.drop_duplicates(subset = 'Ids', keep = 'first', inplace = True)
df.to_csv('Profile.csv', index = False)



@app.route('/', methods=["GET"])
def DetectFace():
    global name
    global name1
    reader = csv.DictReader(open('Profile.csv'))
    print('Detecting Login Faces')
    for rows in reader:
        result = dict(rows)
        print(result)
        if result['Ids'] == '1':
            name1 = result['Name']
        elif result['Ids'] == '2':
            name2 = result["Name"]
    recognizer = cv2.face.LBPHFaceRecognizer_create()  #cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainData\Trainner.yml")

    faceCascade  = cv2.CascadeClassifier('C:/Users/Varun Sai Reddy/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0' \
                  '/LocalCache/local-packages/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Face_Id = ''
    name2 = ''

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        Face_Id = 'Not detected'
 
        for (x, y, w, h) in faces:
            Face_Id = 'Not detected'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 80:
                if Id == 1:
                    name = name1
                    
                elif Id == 2:
                    name = name2
                    
                Predicted_name = str(name)
                Face_Id = Predicted_name
            else:
                Predicted_name = 'Unknown'
                Face_Id = Predicted_name
            
                noOfFile = len(os.listdir("UnknownFaces")) + 1
                if int(noOfFile) < 100:
                    cv2.imwrite("UnknownFaces\Image" + str(noOfFile) + ".jpg", frame[y:y + h, x:x + w])
                
                else:
                    pass


            cv2.putText(frame, str(Predicted_name), (x, y + h), font, 1, (255, 255, 255), 2)
            
        cv2.imshow('Picture', frame)
        #print(Face_Id)
        cv2.waitKey(1)

        if Face_Id == 'Not detected':
            print("-----Face Not Detected, Try again------")
            pass

            
        elif Face_Id == name1 or name2 and Face_Id != 'Unknown' :
           
            print('-----------login successfull-------')
           
          
        
            system = tk.Tk()


            system.geometry('1280x900')


            system.configure(background='black')


            system.resizable(width=False, height=False)


            system.title('Login and Registration System')

            

            system.wm_title('!! CRIMINAL DETECTION AND RECOGNITION !!')


            top_frame = Label(system, text='!! LOGIN SUCCESSFUL !!',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
            top_frame.pack(side='top')

            canvas = Canvas(system, width=500, height=350)
            
            system.mainloop()

            return 'User '+name+' has been logged in successfully'

            break
        else:
            print('-----------Login failed please try again-------')
            return 'User has been logged in successfully'
        
        
        #if (cv2.waitKey(1) == ord('q')):
        #   break
        return 'User has been logged in successfully'

if __name__ == "__main__":
    app.run()

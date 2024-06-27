## Face Recognition based login system


Face login using Open CV Python and Flask framework



### Installation

python 3.8.2

pip install opencv-python
pip install opencv-contrib-python

pip install pillow

pip install pandas

pip install numpy

## Prerequisites

Numpy </br>
OpenCV </br>
Pillow </br>
Pandas</br>


## Process :-

1. Create the same folders : testing or download entire project
2. Run Readface.py : take some images of user for login,id and name needs to be added
3. Run Train.py : this will train all images taken and put in TrainData Folder......
4. Run python Detectface.py to detect the face and Login and login status will be available on browser
5. default server : (localhost:5000)

harcascadePath = 'C:/Users/Varun Sai Reddy/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml'
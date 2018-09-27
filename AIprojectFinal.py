# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:07:11 2018

@author: mrahman8
"""

from tkinter import *
#from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
from tkinter.filedialog import askopenfilename

HomeDirectory="C:/Users/mrahman8/Desktop/Spring 2018/AI/AI project/"
EsarpDirectory="E:/UofM/WorkSpace_Programming/Python/ArtificialIntelligence/"
Directory=HomeDirectory
trainDatabase=Directory+"train.csv"
#data



def readCSV(directory):
    d1=pd.read_csv(trainDatabase)
    data=d1.as_matrix()        
    return data
def trainTheClassifier(data,clf):
    xtrain=data[0:21000,1:] #training dataset
    train_label=data[0:21000,0] 
    x=clf.fit(xtrain,train_label)
    return x

def showAsImage(d):
    #index=5
    #d=xtest[index]
    d.shape=(28,28)
    plt.figure(1)
    img1=d
    plt.imshow(img1,cmap='gray')
    return img1
def saveImage(img1,st):
    savedImage=Directory+st+".png"
    cv.imwrite(savedImage,img1)
    return


def loadImg():
    global img1
    name = askopenfilename(initialdir="C:/Users/mrahman8/Desktop/Spring 2018/AI/AI project/",
                           filetypes =(("Img File", "*.png"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    print (name)
    my_image=PhotoImage(file=name)
    my_imageZ = my_image.zoom(5,5)
    canvas.create_image(0,0,anchor=NW, image=my_imageZ)
    canvas.my_image= my_imageZ
    imageFile=name
    img=cv.imread(imageFile,0)
    #print("From image")
    #print(img[row:row+5,0:10])
    #print("Subtraction result")
    #print(image1-img)
    img1=img
    #Label(image = img1, width=250, height=250).grid(row=1, column=0)
    label1.configure(bg="white")
    label2.configure(bg="white")
    label3.configure(bg="white")
    label4.configure(bg="white")
    


def predictDigit():
    global number
    p1=img1.flatten()
    p2=[p1]
    number=clf.predict(p2)
    print (number)
    label5.configure(text=str(number))
    print ('Prediction:',clf.predict(p2))
    plt.imshow(img1, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

def selectBox():
    
    if 0<= number <= 2:
        label1.configure(text="This Box", bg="green")
    elif 3<= number <= 5:
        label2.configure(text="This Box", bg="green")
    elif 6<= number <= 7:
        label3.configure(text="This Box", bg="green")
    elif 8<= number <= 9:
        label4.configure(text="This Box", bg="green")

#print(d)


 
root = Tk()
frame = Frame(root)
root.title("Automated Letter Sorting Based on Handwritten Digit Recognition")
canvas= Canvas(root, width=300, height=300)
#Learning



#digits = datasets.load_digits()
#features = digits.data 
#labels = digits.target

#clf = SVC(gamma = 0.001, C=100)
#x,y = digits.data[:-10], digits.target[:-10]
#clf.fit(x,y)


data=readCSV(trainDatabase)
#clf=DecisionTreeClassifier() #creating object of the class
clf = SVC(gamma = 0.001, C=100)
x=trainTheClassifier(data,clf)
xtest=data[21000:,1:]
actual_label=data[21000:,0]

 # put the image
#pic1 = "digitPicture.png"
#img1 = ImageTk.PhotoImage(Image.open(pic1))




    




#Label(image = img1,width=250, height=250).grid(row=1, column=0)

label1= Label(text="Box 1 (0-2)", width=50, height=10,bg="white")
label1.grid(row=1, column=2)
Label(width=50, height=2).grid(row=2, column=0)
label2=Label(text="Box 2 (3-5)", width=50, height=10,bg="white")
label2.grid(row=3, column=2)

Label(width=50, height=2).grid(row=4, column=0)
label3=Label(text="Box 3 (6-7)", width=50, height=10,bg="white")
label3.grid(row=5, column=2)
Label(width=50, height=2).grid(row=6, column=0)
label4=Label(text="Box 4 (8-9)", width=50, height=10,bg="white")
label4.grid(row=7, column=2)
label5=Label(width=15, bg="white")
label5.grid(row=1, column=1)

Button(text= "Load the image", command=loadImg, width=20,bg="yellow",fg="black").grid(row=0, column=0)
Button(text= "Predict the no.", command=predictDigit,width=20,bg="yellow").grid(row=0, column=1)
Button(text= "Select the Box", command=selectBox,width=20,bg="yellow").grid(row=0, column=2)
canvas.grid(row=1, column=0)

mainloop()
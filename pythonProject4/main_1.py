import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Button
from  tkinter import messagebox
import random
import os
import sys
from sys import exit
import matplotlib.colors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import metrics

import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

def Signum(value):
    if value <= 0:
        return -1
    else :
        return 1


data= pd.read_csv('PreprocessedData.csv')
X1=data['bill_length_mm']
X2=data['bill_depth_mm']
X3=data['flipper_length_mm']
X4=data['gender']
X5=data['body_mass_g']
Y=data['species']

ADELIE_CLASS=data.iloc[0:50 ,:]
GENTOO_CLASS=data.iloc[50:100 ,:]
CHINSTRAP_CLASS=data.iloc[100:150 ,:]

#Visualization before training
plt.figure('X1 VS X2')
plt.scatter(X1,X2,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')

plt.figure('X1 VS X3')
plt.scatter(X1,X3,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')

plt.figure('X1 VS X4')
plt.scatter(X1,X4,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_length_mm')
plt.ylabel('gender')

plt.figure('X1 VS X5')
plt.scatter(X1,X5,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_length_mm')
plt.ylabel('body_mass_g')

plt.figure('X2 VS X3')
plt.scatter(X2,X3,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_depth_mm')
plt.ylabel('flipper_length_mm')

plt.figure('X2 VS X4')
plt.scatter(X2,X4,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_depth_mm')
plt.ylabel('gender')

plt.figure('X2 VS X5')
plt.scatter(X2,X5,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('bill_depth_mm')
plt.ylabel('body_mass_g')

plt.figure('X3 VS X4')
plt.scatter(X3,X4,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('flipper_length_mm')
plt.ylabel('gender')

plt.figure('X3 VS X5')
plt.scatter(X3,X5,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('flipper_length_mm')
plt.ylabel('body_mass_g')

plt.figure('X4 VS X5')
plt.scatter(X4,X5,c=data['species'],cmap=matplotlib.colors.ListedColormap(['red','blue','black']))
plt.xlabel('gender')
plt.ylabel('body_mass_g')

plt.show()


# Creating tkinter window
window = tk.Tk()
window.title('SLP')
window.geometry('700x680')

# label text for title
ttk.Label(window, text="Single Layer Perceptron :",
          background='red', foreground="white",
          font=("Times New Roman", 12)).grid(row=0, column=1)

# label
ttk.Label(window, text="Select first feature :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=5, padx=10, pady=25)
ttk.Label(window, text="Select second feature :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=6, padx=10, pady=25)
tk.Label(window, text="Select first class :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=7, padx=10, pady=25)
tk.Label(window, text="Select second class :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=8, padx=10, pady=25)
tk.Label(window, text="Epochs :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=9, padx=10, pady=25)

tk.Label(window, text="Learning Rate (eta) :",
          font=("Times New Roman", 12)).grid(column=0,
                                             row=10, padx=10, pady=25)
tk.Label(window, text="Add Bias ",
          font=("Times New Roman", 13)).grid(column=0,
                                             row=11, padx=10, pady=25)


#CheckBox Creation

CBox1 = IntVar()
Checkbutton(window, text="YES",onvalue=1,offvalue=0, variable=CBox1).grid(row=11,column=1, sticky=W,padx=5,pady=0)
CBox2 = IntVar()
Checkbutton(window, text="NO", onvalue=1,offvalue=0,variable=CBox2).grid(row=12, column=1,sticky=W,padx=5,pady=0)


# Combobox creation
n =tk.StringVar()
j=tk.StringVar()
h=tk.StringVar()
t=tk.StringVar()
feat1= ttk.Combobox(window, width=24, textvariable=n)
feat2=ttk.Combobox(window, width=24, textvariable=j)
cla1=ttk.Combobox(window, width=24, textvariable=h)
cla2=ttk.Combobox(window, width=24, textvariable=t)

# Adding combobox drop down list
feat1['values'] = ('bill_length_mm',
                          'bill_depth_mm',
                          'flipper_length_mm',
                          'gender',
                          'body_mass_g',
                        )
feat2['values'] = ('bill_length_mm',
                          'bill_depth_mm',
                          'flipper_length_mm',
                          'gender',
                          'body_mass_g',
                        )
cla1['values'] = ('Adelie',
                          'Gentoo',
                          'Chinstrap',

                        )
cla2['values'] = ('Adelie',
                          'Gentoo',
                          'Chinstrap',

                        )

feat1.grid(column=1, row=5)
feat1.current()
feat2.grid(column=1, row=6)
feat2.current()
cla1.grid(column=1, row=7)
cla1.current()
cla2.grid(column=1, row=8)
cla2.current()

#textbox creation
txt1=Entry(window,width=25)
txt1.place(x="152",y="350")
txt1.focus_set()
txt2=Entry(window,width=25)
txt2.place(x="152",y="421")
txt2.focus_set()



def Training(train_data,Epochs,eta,c,Bias):
    Weights = np.random.random(2)


    for k in range(Epochs):
        # 39.1  18.7   1
        for i,j in zip(train_data[c].values,train_data['species'].values) :

            YPred = Signum(np.dot(np.transpose(Weights),np.array(i))+Bias)

            if YPred != j :
                L= j - YPred
                Weights=(Weights + eta * L * i) +Bias
                if Bias!=0:
                   Bias = Bias + eta * L

    return Weights,Bias
Ypred_List=[]
def Testing (tested_data,Weights,Bias,c):



    Counter1 = 0
    Counter2=0
    for i,j in zip(tested_data[c].values,tested_data['species'].values):


        YPred = Signum(np.dot(np.transpose(Weights), np.array(i)) + Bias)
        Ypred_List.append(YPred)
        if YPred == j and YPred==1:
          Counter1+=1
        elif YPred==j and YPred==-1:
          Counter2 +=1

    L1 = [Counter1,20-Counter1]
    L2 = [20-Counter2,Counter2]

    print("Convution Matrix : ",L1,L2)
    return ((Counter1+Counter2)/len(tested_data))*100

def dline(Weights, Bias,f1,f2,xmax,xmin,data100):

    plt.scatter(data100[f1],data100[f2],c=data100['species'],cmap=matplotlib.colors.ListedColormap(['red','blue']))
    plt.xlabel(f1)
    plt.ylabel(f2)
    #y_values = np.dot((-1. / w2), (np.dot(w1, x_values) + b))
    Y1=((-1*(xmin*Weights[0]+Bias))/Weights[1])
    Y2=((-1*(xmax*Weights[0]+Bias))/Weights[1])
    line_X = [xmin, xmax]
    line_Y = [Y1, Y2]

    plt.plot(line_X,line_Y)
    #plt.plot(xmax, Y2)


    plt.show()

def retrieve_input():
    biasV=0
    Ep=txt1.get()
    eta=txt2.get()
    f1 = feat1.get()

    f2 = feat2.get()

    c1 = cla1.get()
    c2 = cla2.get()

    if len(eta) <= 0 or len(Ep) <= 0 or len(f1) <= 0 or len(f2) <= 0 or len(c1) <= 0 or len(c2) <= 0:
        msg2 = messagebox.showinfo("ERROR", "MISSING DATA !")
    eta_float=float(eta)
    Ep_int=int(Ep)
    ch1=CBox1.get()
    ch2 = CBox2.get()
    if ch1==1 and ch2==0:
        biasV= np.random.random_sample()
    elif ch2==1 and ch1==0:
        biasV=0

    elif (ch1==0 & ch2==0) or (ch1==1 and ch2==1):
        msg=messagebox.showinfo("ERROR","PLEASE CHOOSE ONE OF THE CHEKBOXES !!!!")
        sys.exit()

    if( (c1=='Adelie' and c2=='Adelie') or (c1=='Gentoo' and c2=='Gentoo') or (c1=='Chinstrap' and c2=='Chinstrap')):
        msg3 = messagebox.showinfo("ERROR", "PLEASE CHOOSE DIFFRENT CLASSES !!!!")
        sys.exit()
    if ((f1=='bill_length_mm' and f2=='bill_length_mm') or (f1=='bill_depth_mm' and f2=='bill_depth_mm')  or (f1=='flipper_length_mm' and f2=='flipper_length_mm') or (f1=='gender' and f2=='gender') or (f1=='body_mass_g' and f2=='body_mass_g') ):
        msg5 = messagebox.showinfo("ERROR", "PLEASE CHOOSE DIFFRENT FEATURES !!!!")
        sys.exit()

    if ((c1=='Adelie' or c1=='Gentoo') and (c2=='Gentoo' or c2=='Adelie')) :

        d1train = data.loc[data['species'] == 1][:30]
        d1test = data.loc[data['species'] == 1][30:50]

        d2train = data.loc[data['species'] == 0][:30]

        d2train['species']=d2train['species'].replace(0,-1)
        d2test = data.loc[data['species'] == 0][30:50]

        d2test['species']=d2test['species'].replace(0,-1)
        dataTrain = d1train.append(d2train)
        dataTest = d1test.append(d2test)
        dataTrain = shuffle(dataTrain)
        dataTest = shuffle(dataTest)
        lis=[f1,f2]
        d1train=shuffle(d1train)
        d1test=shuffle(d1test)
        d2train=shuffle(d2train)
        d2test=shuffle(d2test)
        finalData=d1train.append(d1test)
        finalData=finalData.append(d2train)
        finalData=finalData.append(d2test)

        Weight_AG,bi = Training(dataTrain,Ep_int,eta_float,lis,biasV)
        Accuracy=Testing(dataTest,Weight_AG,bi,lis)
        print("Accuracy is ",Accuracy)
        xmin = finalData[f1].min()
        xmax = finalData[f1].max()
        dline(Weight_AG, bi, f1, f2, xmax, xmin,finalData)

    elif ((c1=='Adelie' or c1=='Chinstrap') and (c2=='Chinstrap' or c2=='Adelie')):
        d1train = data.loc[data['species'] == 1][:30]
        d1test = data.loc[data['species'] == 1][30:50]

        d2train = data.loc[data['species'] == -1][:30]
        d2train['species']=d2train['species'].replace(-1,-1)

        d2test = data.loc[data['species'] == -1][30:50]

        d2test['species']=d2test['species'].replace(-1,-1)

        dataTrain = d1train.append(d2train)
        dataTest = d1test.append(d2test)
        dataTrain = shuffle(dataTrain)
        dataTest = shuffle(dataTest)
        lis=[f1,f2]
        d1train = shuffle(d1train)
        d1test = shuffle(d1test)
        d2train = shuffle(d2train)
        d2test = shuffle(d2test)
        finalData = d1train.append(d1test)
        finalData = finalData.append(d2train)
        finalData = finalData.append(d2test)
        Weight_AC,bi = Training(dataTrain,Ep_int,eta_float,lis,biasV)
        Accuracy=Testing(dataTest,Weight_AC,bi,lis)
        print("Accuracy is ",Accuracy)
        # print(CM(Ypred_List, dataTest['species'].values))
        xmin = finalData[f1].min()
        xmax = finalData[f1].max()

        dline(Weight_AC, bi, f1, f2, xmax, xmin,finalData)

    elif ((c1 == 'Gentoo' or c1 == 'Chinstrap') and (c2 == 'Chinstrap' or c2 == 'Gentoo')):
        d1train = data.loc[data['species'] == 0][:30]
        d1test = data.loc[data['species'] == 0][30:50]
        dplot1=d1test
        d1train['species'] = d1train['species'].replace(0, 1)
        d1test['species'] = d1test['species'].replace(0, 1)
        d2train = data.loc[data['species'] == -1][:30]
        d2test = data.loc[data['species'] == -1][30:50]
        dplot2=d2test
        d2train['species'] = d2train['species'].replace(-1, -1)
        d2test['species'] = d2test['species'].replace(-1, -1)
        dataTrain = d1train.append(d2train)
        dataTest = d1test.append(d2test)
        dataTrain = shuffle(dataTrain)
        dataTest = shuffle(dataTest)
        lis = [f1, f2]
        d1train = shuffle(d1train)
        d1test = shuffle(d1test)
        d2train = shuffle(d2train)
        d2test = shuffle(d2test)
        finalData = d1train.append(d1test)
        finalData = finalData.append(d2train)
        finalData = finalData.append(d2test)
        Weight_GC,bi = Training(dataTrain,Ep_int,eta_float,lis,biasV)
        Accuracy=Testing(dataTest,Weight_GC,bi,lis)
        print("Accuracy is ",Accuracy)
        xmin = finalData[f1].min()
        xmax = finalData[f1].max()
        dline(Weight_GC, bi, f1, f2, xmax, xmin,finalData)


#Button Creation
BB=Button(window, height=2, width=20, font=12,text="Submit", fg="white",bg="red",command=retrieve_input).grid(row=17,column=2,pady=9)

window.mainloop()


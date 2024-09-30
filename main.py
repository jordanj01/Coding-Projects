import matplotlib
import numpy as np
import scipy
from PIL.Image import open
import cv2


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

def helper(rep: np.ndarray, der: np.ndarray): #helps make loss derivatives with respect to weight
  #print("rep: " + str(rep))
  #print("der: " + str(der[0]))
  #print("rep shape: " + str(rep.shape))
  #print("der shape: " + str(der[0].shape))
  r = rep
  d = der[0]
  replist = r.tolist()
  derlist = d.tolist()
  l = len(derlist)
  res = np.asarray(replist * l)
  #print(res.shape)
  x = len(replist)
  a = 0; b = x
  for count in range(l):
    res[a:b] = np.dot(res[a:b], d[count])
    a = a + x; b = b + x
  return res

class MLP():

    def __init__(self):
        #Initialize all the parametres
        #Uncomment and complete the following lines
        self.W1= np.random.normal(size=(64, 784), scale=0.1)
        self.b1= np.zeros((64,))
        self.W2= np.random.normal(size=(10, 64), scale=0.1)
        self.b2= np.zeros((10,))
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
      #Feed data through the network
      #Uncomment and complete the following lines
        self.x = x
        self.W1x= np.dot(self.W1, x)
        self.a1= self.W1x + self.b1
        self.f1= sigmoid(self.a1)
        self.W2x= np.dot(self.W2, self.f1)
        self.a2= self.W2x + self.b2
        self.y_hat= softmax(self.a2)
        return self.y_hat

    def update_grad(self,y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
      dA2db2 = np.ones(10) #represents the diagonal of dA2db2
      dA2dW2= self.f1 #non zero elements of dA2dW2
      dA2dF1= self.W2 #not sparse
      dF1dA1= sigmoid(self.a1) * (1 - sigmoid(self.a1)) #nonzero elements of dF1dA1
      dA1db1= np.ones(64) #represents the diagonal of dA1db1
      dA1dW1= self.x #non zero elements of dA1dW1
      dLdA2 = self.y_hat - y #from spec
      dLdW2 = helper(dA2dW2, [dLdA2]) #Might be right
      dLdb2 = dLdA2 * dA2db2 #think this is right
      dLdF1 = np.dot(dLdA2, [dA2dF1]) #think this is right
      dLdA1 = dLdF1 * dF1dA1 #think this is right
      dLdW1 = helper(dA1dW1, dLdA1) #Might be right
      dLdb1 = dLdA1 * dA1db1 #think this is right
      self.W2_grad = self.W2_grad + dLdW2
      self.b2_grad = self.b2_grad + dLdb2
      self.W1_grad = self.W1_grad + dLdW1
      self.b1_grad = self.b1_grad + dLdb1
      pass

    def update_params(self,learning_rate):
      self.W2 = self.W2 - learning_rate * self.W2_grad.reshape(10,64)
      self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(10)
      self.W1 = self.W1 - learning_rate * self.W1_grad.reshape(64,784)
      self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)

import matplotlib
matplotlib.use('Agg')
import tkinter as tk
from PIL import ImageTk,Image,ImageDraw

def event_function(event):
    
    x=event.x
    y=event.y
    
    x1=x-20
    y1=y-20
    
    x2=x+20
    y2=y+20

    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')

def clear():
    
    global img,img_draw
    
    canvas.delete('all')
    img=Image.new('RGB',(500,500),(0,0,0))
    img_draw=ImageDraw.Draw(img)    

def predict():
    img_array=np.array(img)
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array=cv2.resize(img_array,(28,28))
    
    img_array=img_array/255.0
    img_array=img_array.reshape((784,))
    result=mlp.forward(img_array)
    x = np.argmax(result)
    label.config(text="Prediction: " + str(x))

mlp = MLP()
mlp.W1 = np.load('./lastMLPW1.npy')
mlp.b1 = np.load('./lastMLPb1.npy')    
mlp.W2 = np.load('./lastMLPW2.npy')
mlp.b2 = np.load('./lastMLPb2.npy')
draw = tk.Tk()

img=Image.new('RGB',(500,500),(0,0,0))
img_draw=ImageDraw.Draw(img)

label = tk.Label(draw, text="Prediction: N/A", bg='white', font="Helvetica 24 bold")
label.grid(row=2, column=0, columnspan=4)

canvas=tk.Canvas(draw,width=500,height=500,bg='white')
canvas.grid(row=0,column=0,columnspan=4)
canvas.bind('<B1-Motion>',event_function)

button_predict=tk.Button(draw,text='PREDICT',bg='blue',fg='white',font='Helvetica 20 bold',command=predict)
button_predict.grid(row=1,column=1)

button_clear=tk.Button(draw,text='CLEAR',bg='yellow',fg='white',font='Helvetica 20 bold',command=clear)
button_clear.grid(row=1,column=2)

button_exit=tk.Button(draw,text='EXIT',bg='red',fg='white',font='Helvetica 20 bold',command=draw.destroy)
button_exit.grid(row=1,column=3)

draw.mainloop()

import threading
import cv2
import PIL
import numpy as np
from PIL import Image, ImageTk
from tkinter import *
from CNN import CNN


def call_translate():
     threading.Timer(2.0, call_translate).start()
     translate(segmented_image)


def translate(img):
    global var
    cnn = CNN()
    latter = cnn.test_convolution_neural_network(img)
    var.set(latter)


def canny_edge_detection(hand_image):
    edges = cv2.Canny(hand_image, 150, 150)
    image_x, image_y = 150, 150
    resized_image = cv2.resize(edges, (image_x, image_y))
    return edges, resized_image


def hsv(hand_image):
    # lower_blue = np.array([0, 150, 50])
    # upper_blue = np.array([195, 255, 255])
    lower_blue = np.array([0, 20, 70])
    upper_blue = np.array([20, 255, 255])

    hsv_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    return mask


def show_frame():
    global segmented_image
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    hand_image = img[102:298, 427:623]
    image_x, image_y = 150, 150
    resized_image = cv2.resize(hand_image, (image_x, image_y))

    edges, segmented_image = canny_edge_detection(resized_image)

    # translate(segmented_image)

    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(hand_image)
    imgtk = ImageTk.PhotoImage(image=img)
    hand.imgtk = imgtk
    hand.configure(image=imgtk)

    img = PIL.Image.fromarray(edges)
    imgtk = ImageTk.PhotoImage(image=img)
    segmented_hand.imgtk = imgtk
    segmented_hand.configure(image=imgtk)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera.imgtk = imgtk
    camera.configure(image=imgtk)
    camera.after(1, show_frame)


root = Tk()

width, height = 800, 600
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

segmented_image = cv2.imread('A.12.jpg', 0)

camera = Label(root)
hand = Label(root)
segmented_hand = Label(root)

var = StringVar()
label = Label(root, anchor=CENTER, bg="#8B0000",fg="#FFFFFF",bd="5", textvariable=var,
              relief=FLAT, text="Helvetica", font=("Times", 86,"bold") , padx="20")
var.set(" * ")

show_frame()
call_translate()
camera.pack()
hand.pack(side=LEFT)
segmented_hand.pack(side=RIGHT)
label.pack()

root.mainloop()

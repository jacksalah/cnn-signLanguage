import cv2
import os


def rename_pictures():
    directory = "D:\\computer science\\graduation project\\midyear presentation\\prototype\\sandra\\ProtoType\\training_set\\Z"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        word_rank = src.split('.')[0]
        dist =  "Z." + word_rank + ".png"
        os.rename(src, dist)
        i += 1


def segment():
    directory = "D:\\computer science\\graduation project\\midyear presentation\\prototype\\sandra\\ProtoType\\TrainData"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        word_label = src.split('.')[0]
        word_rank = src.split('.')[1]

        img = cv2.imread(src, 0)
        edges = cv2.Canny(img, 64, 64)
        dist = word_label + "." + word_rank
        cv2.imwrite(dist + ".png", edges)
        i += 1


def delete():
    directory = "E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_black\\train - Copy"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        word_label = src.split('.')[2]
        if word_label == "jpeg":
            os.remove("E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_black\\train - Copy\\" + src)
        i += 1

segment()
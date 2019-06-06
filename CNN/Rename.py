import cv2
import os


def rename_pictures():
    directory = "E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\A"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        # word_label = src.split('.')[0]
        word_rank = src.split('.')[0]
        dist = "A." + word_rank + ".png"
        os.rename(src, dist)
        i += 1


def segment():
    directory = "E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\test"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        word_label = src.split('.')[0]
        word_rank = src.split('.')[1]

        img = cv2.imread(src, 0)
        edges = cv2.Canny(img, 150, 150)
        dist = word_label + "." + word_rank
        cv2.imwrite(dist + ".png", edges)
        i += 1


def delete():
    directory = "E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\test"
    os.chdir(directory)
    i = 1
    for file in os.listdir():
        src = file
        word_label = src.split('.')[2]
        if word_label == "jpg":
            os.remove("E:\\4th year\\1st term\\Project\\prototype\\CNN\\dataset_canny\\test_set\\test\\" + src)
        i += 1


segment()
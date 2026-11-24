import cv2 as cv
import os
import sys
# from ultralytics import YOLO
import numpy as np
import argparse
def Creat_mask_by_inRange(hsv,color):
    if color=="orange":
        lower = np.array([10, 76, 204])
        upper = np.array([40, 255, 255])
    elif color=="blue":
        lower = np.array([100, 100, 178])
        upper = np.array([130, 255, 255])
    elif color=="red":
        lower = np.array([160, 150, 150])
        upper = np.array([180, 255, 255])
        mask1=cv.inRange(hsv, lower, upper)
        lower = np.array([0, 150, 108])
        upper = np.array([10, 255, 255])
        mask2=cv.inRange(hsv, lower, upper)
        return cv.bitwise_or(mask1, mask2)
    mask=cv.inRange(hsv, lower, upper)
    return mask
def Blur_img(img,way="Gaussian",kernel_size=3):
    ksize = (kernel_size, kernel_size)
    if way=="Gaussian":
        return cv.GaussianBlur(img, ksize, 0)
    elif way=="Median":
        return cv.medianBlur(img, kernel_size)
    elif way=="Bilateral":
        return cv.bilateralFilter(img, kernel_size, 75, 75)
    elif way=="blur":
        return cv.blur(img, ksize)
    elif way=="None":
        return img
def Close_img(img,kernel_size=3):
    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    return cv.morphologyEx(
    img,
    cv.MORPH_CLOSE,
    kernel
    )
def Open_img(img,kernel_size=3):
    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    return cv.morphologyEx(
    img,
    cv.MORPH_OPEN,
    kernel
    )
def Get_mask(img):
    Blurred_img=Blur_img(img,way="Gaussian")
    hsv = cv.cvtColor(Blurred_img, cv.COLOR_BGR2HSV)
    mask=cv.bitwise_or(Creat_mask_by_inRange(hsv,"red"),Creat_mask_by_inRange(hsv,"orange"))
    mask_closed=Close_img(mask,5)
    # mask_open=Open_img(mask_closed,1)
    return mask_closed
def check_light(rect):
    # print(rect)
    # sys.exit()
    width=min(rect[1][1],rect[1][0])
    height=max(rect[1][1],rect[1][0])
    area=width*height
    # print(width,height)
    if area<20:
        return 0
    if float(height)/float(width)<2:
        return 0
    return 1
def find_light(mask):
    contours, hierarchy = cv.findContours(
    mask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
    )
    lights=[]
    for contour in contours:
        rect = cv.minAreaRect(contour)
        if not check_light(rect):
            continue
        lights.append(rect)
    return lights
def get_angle(rect):
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle += 90
    return angle % 180
def check_match(light1,light2):
    width1=min(light1[1][1],light1[1][0])
    height1=max(light1[1][1],light1[1][0])
    x1=light1[0][0]
    y1=light1[0][1]
    angle1=get_angle(light1)
    width2=min(light2[1][1],light2[1][0])
    height2=max(light2[1][1],light2[1][0])
    x2=light2[0][0]
    y2=light2[0][1]
    angle2=get_angle(light2)
    if abs(angle1-angle2)>20:
        return 0
    if float(abs(height1-height2))/float(height1)>0.3:
        return 0
    if float(abs(y1-y2))>0.5*height1:
        return 0
    if float(abs(x1-x2))>5*height1:
        return 0
    if float(abs(x1-x2))/float(height1)<1.0 or float(abs(x1-x2))/float(height1)>5.0:
        return 0
    return 1
def sort_points(points):
    order_x = np.argsort(points[:, 0])
    order_y = np.argsort(points[:, 1])
    top=order_y[:2]
    bottom=order_y[2:]
    if points[top[0]][0]<points[top[1]][0]:
        left_top=top[0]
        right_top=top[1]
    else:
        left_top=top[1]
        right_top=top[0]
    if points[bottom[0]][0]<points[bottom[1]][0]:
        left_bottom=bottom[0]
        right_bottom=bottom[1]
    else:
        left_bottom=bottom[1]
        right_bottom=bottom[0]
    return np.array([
        points[left_top],
        points[left_bottom],
        points[right_top],
        points[right_bottom]
    ])
def Get_aim(light1,light2):
    points1 = sort_points(np.int32(cv.boxPoints(light1)))
    points2 = sort_points(np.int32(cv.boxPoints(light2)))
    if light1[0][0]<light2[0][0]:
        aim_points=np.array([points1[0],points2[2],points2[3],points1[1]],np.int32)
    else:
        aim_points=np.array([points2[0],points1[2],points1[3],points2[1]],np.int32)
    return aim_points
def Match_light(lights):
    array_aim_points=[]
    for i in range(len(lights)):
        for j in range(i+1,len(lights)):
            if check_match(lights[i],lights[j]):
                array_aim_points.append(Get_aim(lights[i],lights[j]))
    return array_aim_points
def Dispose_image_inRange(img):
    mask=Get_mask(img)
    lights=find_light(mask)
    array_aim_points= Match_light(lights)
    return array_aim_points
# def Dispose_yolo(img,result):
#     global model
#     results = model.predict(img, conf=0.5,device=0,verbose=False)
#     boxes = results[0].boxes
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         cv.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     pass
# def Dispose_video(file_path="./input/video.avi"):
#     cap=cv.VideoCapture(file_path)
#     fourcc = cv.VideoWriter_fourcc(*"XVID")
#     out = cv.VideoWriter(
#     "Output_video.avi",
#     fourcc,
#     30,
#     (1280, 1024)
#     )
#     if not cap.isOpened():
#         print("error")
#         return
#     while True:
#         ret,img=cap.read()
#         if not ret:
#             break
#         result=img.copy()
#         Dispose_yolo(img,result)
#         out.write(result)
#         cv.imshow("img",result)
#         cv.waitKey(1)
#     out.release()
#     cap.release()
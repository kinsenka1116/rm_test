import cv2 as cv
import os
import sys
import numpy as np
import argparse

def Input_image(image_path=None):
    if image_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", type=str, default="./images/outpost_frame.png")
        image_path=parser.parse_args().image_path
    img=cv.imread(image_path)
    if img is None:
        print("error")
        sys.exit(0)
    return img
def Out_image(img,filename):
    os.makedirs("outputs", exist_ok=True)
    if not cv.imwrite(f"outputs/{filename}.png",img):
        print("error")
        sys.exit(0)
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
def Get_mask(img,color):
    Blurred_img=Blur_img(img,way="Gaussian")
    hsv = cv.cvtColor(Blurred_img, cv.COLOR_BGR2HSV)
    if color=="orange":
        mask=cv.bitwise_or(Creat_mask_by_inRange(hsv,"red"),Creat_mask_by_inRange(hsv,"orange"))
    else:
        mask=Creat_mask_by_inRange(hsv,color)
    Out_image(Creat_mask_by_inRange(hsv,"blue"),"mask_blue")
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
def find_light(mask,result_light):
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
        points = np.intp(cv.boxPoints(rect))
        lights.append(rect)
        cv.drawContours(result_light, [points], 0, (0, 255, 0), 2)
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
    # x_max=max(points[:,0])
    # x_min=min(points[:,0])
    # y_max=max(points[:,1])
    # y_min=min(points[:,1])
    # return np.array([
    #     (x_min,y_min),(x_min,y_max),(x_max,y_min),(x_max,y_max)
    # ])
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
def print_aim(aim_points,color):
    M = cv.moments(aim_points)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    with open("./outputs/information.txt", "w", encoding="utf-8") as f:
        f.write(f"\t中心点:({int(cx)},{int(cy)})\n")
        f.write(f"\t颜色:{color}\n")
        f.write(f"\t坐标:左上:{aim_points[0]} 左下:{aim_points[1]} 右下:{aim_points[2]} 右上:{aim_points[3]}\n")
        f.write(f"\n")
def putText(point,text,result):
    cv.putText(
    result,
    text,
    point,
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
    )
def make_text(aim_points,result):
    putText(aim_points[0],"左上",result)
    putText(aim_points[1],"左下",result)
    putText(aim_points[2],"右下",result)
    putText(aim_points[3],"右上",result)
def solvePnP(aim_points):
    pass
def draw_aim(light1,light2,result,color):
    points1 = sort_points(np.int32(cv.boxPoints(light1)))
    points2 = sort_points(np.int32(cv.boxPoints(light2)))
    if light1[0][0]<light2[0][0]:
        aim_points=np.array([points1[0],points1[1],points2[3],points2[2]],np.int32)
    else:
        aim_points=np.array([points2[0],points2[1],points1[3],points1[2]],np.int32)
    # print(len(aim_points))
    print_aim(aim_points,color)
    make_text(aim_points,result)
    cv.drawContours(result, [aim_points], 0, (0, 255, 0), 2)
    solvePnP(aim_points)
    pass
def Match_light(lights,result,color):
    for i in range(len(lights)):
        for j in range(i+1,len(lights)):
            if check_match(lights[i],lights[j]):
                draw_aim(lights[i],lights[j],result,color)
def Dispose_image(img,result,color="orange"):
    mask=Get_mask(img,color)
    Out_image(mask,"mask")
    result_light=img.copy()
    lights=find_light(mask,result_light)
    Out_image(result_light,"light")
    # cv.imshow("mask",mask)
    # print(len(lights))
    Match_light(lights,result,color)
def Dispose_video(file_path="./input/video.avi"):
    cap=cv.VideoCapture(file_path)
    if not cap.isOpened():
        print("error")
        return
    while True:
        ret,img=cap.read()
        if not ret:
            break
        result=img.copy()
        Dispose_image(img,result)
        cv.imshow("img",result)
        cv.waitKey(1)
    cap.release()
def main():
    img=Input_image()
    result=img.copy()
    Dispose_image(img,result)
    # cv.imshow("img",result)
    Out_image(result,"result")
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()
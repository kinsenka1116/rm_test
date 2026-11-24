import cv2 as cv
import os
import sys
import numpy as np
import argparse
import Const
import math
import input
import output
import projectPoints
import Find_Aim
import Solve_PnP
def image_process(frame_id, time_s, img):
    global filename
    img=cv.resize(img,(Const.image_width, Const.image_height))
    array_aim_points=Find_Aim.Dispose_image_inRange(img)
    if len(array_aim_points)==0:
        output.Find_None_Aim(frame_id, time_s)
        return img
    array_ret, array_rvec, array_tvec = Solve_PnP.PnP_img(array_aim_points)
    
    
    p=array_aim_points[0][1][0]-array_aim_points[0][0][0]
    Z_rough=Const.K[0,0]*Const.w/(p)
    tz=array_tvec[0][2][0]
    delta=abs(Z_rough-tz)/tz*100


    print(filename)
    print(f"p={p}")
    print(f"Z_rough={Z_rough}   tz={tz} delta={delta}%")



    result=img.copy()
    array_imagePoints=projectPoints.project_points_img(array_ret, array_rvec, array_tvec, result)
    output.output_frame_csv(frame_id, time_s, array_aim_points, array_ret, array_rvec, array_tvec,array_aim_points,array_imagePoints)
    # result=cv.drawContours(result, array_aim_points, -1, (0, 255, 0), 3)
    return result
    # cv.imshow("result", result)
    # cv.waitKey(0)
if __name__=="__main__":
    filename="test_2.png"
    result=image_process(1, 0, input.Input_image(f"./input/{filename}"))
    # cv.imshow("result", result)
    cv.waitKey(0)
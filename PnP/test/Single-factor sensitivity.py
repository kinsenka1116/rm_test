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
def Get_reprojection_error(aim_points, imagePoints):
    if imagePoints is None:
        return None
    error = 0.0
    for i in range(len(aim_points)):
        u_i, v_i = aim_points[i]
        u_hat_i, v_hat_i = imagePoints[i]
        e_i = ((u_i - u_hat_i) ** 2 + (v_i - v_hat_i) ** 2) ** 0.5
        error += e_i
    reprojection_error_px = error / len(aim_points)
    return reprojection_error_px
def image_process(frame_id, time_s, img):
    global filename

    # Const.K[0,0]*=2.0
    # Const.K[1,1]*=2.0

    # Const.w*=1.10
    # Const.objectPoints=np.array([
    # [-Const.w / 2, -Const.h / 2, 0],
    # [ Const.w / 2, -Const.h / 2, 0], 
    # [ Const.w / 2,  Const.h / 2, 0],
    # [-Const.w / 2,  Const.h / 2, 0]
    # ],dtype=np.float64)

    img=cv.resize(img,(Const.image_width, Const.image_height))
    array_aim_points=Find_Aim.Dispose_image_inRange(img)
    if len(array_aim_points)==0:
        output.Find_None_Aim(frame_id, time_s)
        return img
    # array_aim_points[0][3]+=10
    array_ret, array_rvec, array_tvec = Solve_PnP.PnP_img(array_aim_points)
    result=img.copy()
    array_imagePoints=projectPoints.project_points_img(array_ret, array_rvec, array_tvec, result)


    tx=array_tvec[0][0][0]
    ty=array_tvec[0][1][0]
    tz=array_tvec[0][2][0]
    distance=math.sqrt(tx**2+ty**2+tz**2)
    reprojection_error=Get_reprojection_error(array_aim_points[0],array_imagePoints[0])
    # print(array_rvec[0])
    print(tz,distance,reprojection_error)

    
    # output.output_frame_csv(frame_id, time_s, array_aim_points, array_ret, array_rvec, array_tvec,array_aim_points,array_imagePoints)
    # result=cv.drawContours(result, array_aim_points, -1, (0, 255, 0), 3)
    return result
    # cv.imshow("result", result)
    # cv.waitKey(0)
if __name__=="__main__":
    filename="test_3.png"
    result=image_process(1, 0, input.Input_image(f"./input/{filename}"))
    # cv.imshow("result", result)
    cv.waitKey(0)
import cv2 as cv
import os
import sys
import numpy as np
import argparse
import Const
import input
import PnP.output as output
import PnP.projectPoints as projectPoints
import PnP.Find_Aim as Find_Aim
import PnP.tools as tools
import PnP.Solve_PnP as Solve_PnP
def image_process(img):
    img=cv.resize(img,(Const.image_width, Const.image_height))
    array_aim_points=Find_Aim.Dispose_image_inRange(img)
    if len(array_aim_points)==0:
        return -1
    #存在目标
    array_ret, array_rvec, array_tvec = Solve_PnP.PnP_img(array_aim_points)
    array_imagePoints=projectPoints.project_points_img(array_ret, array_rvec, array_tvec)
    #只处理下标为0的目标
    ret,aim_points,imagePoints=array_ret[0],array_aim_points[0],array_imagePoints[0]
    rvec,tvec=array_rvec[0],array_tvec[0]
    center=np.mean(aim_points,axis=0)
    reprojection_error=tools.Get_reprojection_error(aim_points,imagePoints)
    if ret:#PnP有效
        PnP_state=tools.check_PnP(aim_points,imagePoints,tvec[2][0])
        if PnP_state==0:
            return aim_points,center,tvec[:,0],reprojection_error
        else :
            return PnP_state
    else :#PnP无效
        return 1
#返回单个数字代表计算失败，返回一个元组代表计算成功
#-1 ：找不到目标
# 0：计算成功，返回元组
#1 ：PnP失败
#2 ：PnP成功，但是重投影误差过大
#3 ：PnP成功，但tvec_z<0
#4 :PnP成功，但数量级不对
#frame_id,time_s,candidate_id=0,[aim_points],[center_u_px,center_v_px],plate_id,pnp_valid,tvec_x_m,tvec_y_m,tvec_z_m,reprojection_error_px,used_for_state,used_for_distance_filter,distance_raw_m,distance_filtered_m,reason
def video_process(video_path):
    cap=cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(
    "./outputs/Output_video.avi",
    fourcc,
    fps,
    (Const.image_width, Const.image_height)
    )
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    output.Create_csv()
    frame_id=1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_s=cap.get(cv.CAP_PROP_POS_MSEC)/1000.0
        result=image_process(frame_id, time_s, frame)
        out.write(result)
        frame_id+=1
    cap.release()
    out.release()
    cv.destroyAllWindows()
# if __name__=="__main__":
#     video_process("./input/video.avi")
    # result=image_process(1, 0, input.Input_image("./input/img_0.png"))
    # cv.imshow("result", result)
    # cv.waitKey(0)
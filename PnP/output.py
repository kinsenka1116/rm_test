import cv2 as cv
import Const
import os
import csv
def Out_image(img,filename):
    os.makedirs("outputs", exist_ok=True)
    cv.imwrite(f"outputs/{filename}.png",img)
def Create_csv():
    with open("./outputs/data.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "time_s", "candidate_id",
                         "detect_valid", "pnp_valid",
                         "tx_m", "ty_m", "tz_m", "distance_m",
                         "rx_rad", "ry_rad", "rz_rad", "reprojection_error_px"])
def Find_None_Aim(frame_id, time_s):
    with open("./outputs/data.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        list_row = []
        list_row.append(frame_id)
        list_row.append(time_s)
        list_row.append(False)
        writer.writerow(list_row)
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
def check_PnP(ret,aim_points,imagePoints,tvec):
    p=aim_points[1][0]-aim_points[0][0]
    Z_rough=Const.K[0,0]*Const.w/(p)
    tz=tvec
    delta=abs(Z_rough-tz)/min(tz, Z_rough)
    if ret and Get_reprojection_error(aim_points, imagePoints)<=5.0 and tvec>=0 and delta<10:
        return 1
    else :
        return 0
def output_frame_csv(frame_id, time_s,array_detect, array_ret, array_rvec, array_tvec,array_aim_points,array_imagePoints):
    with open("./outputs/data.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for i in range(len(array_ret)):
            list_row = []
            list_row.append(frame_id)
            list_row.append(time_s)
            list_row.append(i)
            list_row.append(True)
            if check_PnP(array_ret[i],array_aim_points[i], array_imagePoints[i], array_tvec[i][2][0]):
                list_row.append(True)
                list_row.append(array_tvec[i][0][0])
                list_row.append(array_tvec[i][1][0])
                list_row.append(array_tvec[i][2][0])
                distance = (array_tvec[i][0][0]**2 + array_tvec[i][1][0]**2 + array_tvec[i][2][0]**2)**0.5
                list_row.append(distance)
                list_row.append(array_rvec[i][0][0])
                list_row.append(array_rvec[i][1][0])
                list_row.append(array_rvec[i][2][0])
                list_row.append(Get_reprojection_error(array_aim_points[i], array_imagePoints[i]))
            else :
                list_row.append(False)
            writer.writerow(list_row)
# frame_id, time_s, candidate_id,
# detect_valid, pnp_valid,
# tx_m, ty_m, tz_m, distance_m,
# rx_rad, ry_rad, rz_rad, reprojection_error_px

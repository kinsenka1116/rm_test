import numpy as np
import Const
import cv2 as cv
def Same_aim(point_a,point_b):
    distance = np.linalg.norm(point_a - point_b)
    if distance<3*Const.w:
        return 1
    else :
        return 0
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
def check_fitting(data,fps):
    #要求三个轴向层级各有至少 20 个通过质量检查的 PnP 观测；每层连续相位跨度不少于 90°（还没拟合，不知道怎么求跨度...）；拟合时间跨度不少于 0.5 s
    continuous_time=(len(data[0])+len(data[1])+len(data[2]))/fps
    if len(data[0])>=20 and len(data[1])>=20 and len(data[2])>=20 and continuous_time>=0.5:
        return 1
    else :
        return 0
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
def projectPoints(coordinate,rvec,tvec):
    # print(coordinate)
    coordinate=np.array(coordinate,dtype=np.float64)
    coordinate = coordinate.reshape(1, 1, 3)
    imagePoints, _ = cv.projectPoints(coordinate,rvec,tvec,Const.K,Const.D)
    return (imagePoints.reshape(-1, 2))[0]
import cv2 as cv
import os
import Const
import numpy as np
def PnP_img(array_aim_points):
    array_ret=[]
    array_rvec=[]
    array_tvec=[]
    for aim_points in array_aim_points:
        ret,rvec,tvec=solve_pnp(aim_points)
        array_ret.append(ret)
        array_rvec.append(rvec)
        array_tvec.append(tvec)
    return array_ret, array_rvec, array_tvec
def solve_pnp(aim_points):
    objectPoints=Const.objectPoints
    imagePoints = np.asarray(aim_points, dtype=np.float64)
    ret, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, Const.K, Const.D)
    return ret,rvec,tvec
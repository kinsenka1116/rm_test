import cv2 as cv
import numpy as np
import Const
def project_points_img(array_ret, array_rvec, array_tvec):
    array_imagePoints=[]
    for i in range(len(array_ret)):
        if array_ret[i]:
            imagePoints=project_points(Const.objectPoints, array_rvec[i], array_tvec[i]).reshape(-1, 2).astype(np.int32)
            array_imagePoints.append(imagePoints)
        else:
            array_imagePoints.append(None)
    # print("Projected points on image:", array_imagePoints)
    return array_imagePoints
def project_points(objectPoints, rvec, tvec):
    imagePoints, _ = cv.projectPoints(objectPoints, rvec, tvec, Const.K, Const.D)
    return imagePoints
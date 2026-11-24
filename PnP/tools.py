import Const
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
def check_PnP(aim_points,imagePoints,tvec_z):
    p=aim_points[1][0]-aim_points[0][0]
    Z_rough=Const.K[0,0]*Const.w/(p)
    tz=tvec_z
    delta=abs(Z_rough-tz)/min(tz, Z_rough)
    if Get_reprojection_error(aim_points, imagePoints)>5.0:
        return 2  
    if  tvec_z<0 :
        return 3
    if delta>10 :
        return 4
    return 0
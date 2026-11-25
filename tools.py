import numpy as np
import Const
def Same_aim(point_a,point_b):
    distance = np.linalg.norm(point_a - point_b)
    if distance<2*Const.w:
        return 1
    else :
        return 0
def check_fitting(data,fps):
    #要求三个轴向层级各有至少 10 个通过质量检查的 PnP 观测；每层连续相位跨度不少于 90°（还没拟合，不知道怎么求跨度...）；拟合时间跨度不少于 0.5 s
    continuous_time=(len(data[0])+len(data[1])+len(data[2]))/fps
    if len(data[0])>=10 and len(data[1])>=10 and len(data[2])>=10 and continuous_time>=0.5:
        return 1
    else :
        return 0
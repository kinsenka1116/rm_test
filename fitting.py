import numpy as np
import Const
import cv2 as cv
import numpy as np
import tools
import math
from fit_circle_fixed_radius import fit_circle_fixed_radius
class model:
    def __init__(self,data,data_time):
        self.data=data
        self.data_time=data_time
        self.states=1
        self.mode=1
        self.omega=0
        self.theta0=0
        self.e0,self.e90=None,None
        self.mid_id=None
        self.aims=[]
        self.x_array,self.y_array=[],[]
        self.C,self.a=None,None
    def fit_circle_3d(self,data: list) -> tuple[np.ndarray, float, np.ndarray]:
        points = np.asarray(data, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("data 至少要包含 3 个点，每个点有 x、y、z 三个坐标")
        if not np.isfinite(points).all():
            raise ValueError("坐标必须是有限数值")

        # 以平均点为原点，缩放坐标以改善数值计算。
        origin = points.mean(axis=0)
        centered = points - origin
        scale = np.max(np.abs(centered))
        if scale == 0:
            raise ValueError("所有点重合，无法拟合圆")
        centered = centered / scale

        # u、v 是平面内的两个单位方向，normal 垂直于平面。
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        u, v, normal = vh
        x = centered @ u
        y = centered @ v

        # 将圆方程展开为 2*a*x + 2*b*y + c = x**2 + y**2。
        matrix = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
        solution, _, rank, _ = np.linalg.lstsq(matrix, x**2 + y**2, rcond=None)
        if rank < 3:
            raise ValueError("点共线或太接近共线，无法确定圆")

        # 将二维圆心还原到三维坐标，并恢复原来的尺度。
        a, b, c = solution
        center = origin + scale * (a * u + b * v)
        radius = float(scale * np.sqrt(a**2 + b**2 + c))

        # 两个相反的法向量都合法。这里让绝对值最大的分量为正。
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal

        return center, radius, normal
    def fit_parallel_direction(self,vectors: np.ndarray) -> np.ndarray:
        """将 N 个近似平行的三维向量拟合成一个三维单位向量。"""
        vectors = np.asarray(vectors, dtype=np.float64)

        if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] == 0:
            raise ValueError("vectors 必须是形状为 (N, 3) 的非空数组")
        if not np.isfinite(vectors).all():
            raise ValueError("向量分量必须是有限实数")

        # 先缩放，再归一化，避免分量过大或过小时求长度出错。
        scales = np.max(np.abs(vectors), axis=1, keepdims=True)
        if np.any(scales == 0):
            raise ValueError("零向量没有方向")

        scaled = vectors / scales
        unit_vectors = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)

        # 取最大特征值对应的单位特征向量，作为拟合轴。
        _, eigenvectors = np.linalg.eigh(unit_vectors.T @ unit_vectors)
        direction = eigenvectors[:, -1]

        # 根据点积判断朝向，选择多数向量指向的一端。
        # 两边数量相同时，以第一个向量为准。
        projections = unit_vectors @ direction
        same = np.count_nonzero(projections > 0)
        opposite = np.count_nonzero(projections < 0)

        if opposite > same or (opposite == same and projections[0] < 0):
            direction = -direction

        return direction
    def find_middle_point(self,a, b, c):
        errors = [
            np.linalg.norm(a - (b + c) / 2),  # a 到 b、c 中点的距离
            np.linalg.norm(b - (a + c) / 2),  # b 到 a、c 中点的距离
            np.linalg.norm(c - (a + b) / 2),  # c 到 a、b 中点的距离
        ]
        return np.argmin(errors)
    def fit_line(self,array_x: list[float], array_y: list[float]) -> tuple[float, float]:
        x = np.asarray(array_x, dtype=float)
        y = np.asarray(array_y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
            raise ValueError("两个列表必须是一维、等长的，且至少包含两个点")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("坐标必须是有限数值")

        # 按 x 从小到大排列，同时保持每个 x 和 y 的对应关系
        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]

        if x[0] == x[-1]:
            raise ValueError("所有 x 坐标相同，无法确定斜率")

        # 先展开相位，再对展开后的数据做一次拟合
        phase = np.unwrap(y)
        k, b_at_x0 = np.polyfit(x - x[0], phase, deg=1)

        # 将横坐标原点从 x[0] 换回 0，得到 y = kx + b 的截距
        b = b_at_x0 - k * x[0]
        return float(k), float(b)
    def update_e0_e90(self) -> None:
        """根据旋转轴 self.a 计算并保存 self.e0 和 self.e90。"""
        a = np.asarray(self.a, dtype=float).reshape(3)
        a_squared = np.dot(a, a)
        if a_squared == 0:
            raise ValueError("旋转轴 self.a 不能是零向量")

        # 将 x 轴方向投影到垂直于 a 的平面
        ref = np.array([1.0, 0.0, 0.0])
        projected = ref - (np.dot(ref, a) / a_squared) * a

        # 投影太短时，改用 z 轴方向
        if np.linalg.norm(projected) < 1e-6:
            ref = np.array([0.0, 0.0, 1.0])
            projected = ref - (np.dot(ref, a) / a_squared) * a

        self.e0 = projected / np.linalg.norm(projected)
        self.e90 = np.cross(a, self.e0)
    def check_R(self):
        #R=0.275 m
        if Const.R-0.5<self.aims[0][1]<Const.R+0.5 and Const.R-0.5<self.aims[1][1]<Const.R+0.5 and Const.R-0.5<self.aims[2][1]<Const.R+0.5:
            return
        else:
            self.states=0
    def check(self):
        pass
    def Get_theta_mid(self,t):
        theta_mid=self.omega*t+self.theta0
        return tools.normalize_angle(theta_mid)
    def Predict_position(self,i,t):
        #P_i = C + h_i * a+ R * [e0 * cos(theta + delta_i) + e90 * sin(theta + delta_i)]
        theta_mid=self.Get_theta_mid(t)
        if i=="minus" :
            h_i=-Const.interval
            delta_i=self.mode*2*np.pi/3
        elif i=="middle":
            h_i=0
            delta_i=0
        elif i=="plus":
            h_i=Const.interval
            delta_i=self.mode*2*np.pi/3*(-1)
        P_i = self.C + h_i * self.a+ Const.R * (self.e0 * math.cos(theta_mid + delta_i) + self.e90 * math.sin(theta_mid + delta_i))
        return P_i
    def chose_model(self):


        self.mode = 1
        sum_1, N_1 = 0, 0
        i=self.minus_id
        for j in range(len(self.data[i])):
            p_k = self.data[i][j]
            t = self.data_time[i][j]
            N_1 += 1
            P_i = self.Predict_position("minus", t)
            sum_1 += np.dot(p_k - P_i, p_k - P_i)
        i=self.plus_id
        for j in range(len(self.data[i])):
            p_k = self.data[i][j]
            t = self.data_time[i][j]
            N_1 += 1
            P_i = self.Predict_position("plus", t)
            sum_1 += np.dot(p_k - P_i, p_k - P_i)
        RMSE_1 = math.sqrt(sum_1 / N_1)


        self.mode = -1
        sum_2, N_2 = 0, 0
        i=self.minus_id
        for j in range(len(self.data[i])):
            p_k = self.data[i][j]
            t = self.data_time[i][j]
            N_2 += 1
            P_i = self.Predict_position("minus", t)
            sum_2 += np.dot(p_k - P_i, p_k - P_i)
        i=self.plus_id
        for j in range(len(self.data[i])):
            p_k = self.data[i][j]
            t = self.data_time[i][j]
            N_2 += 1
            P_i = self.Predict_position("plus", t)
            sum_2 += np.dot(p_k - P_i, p_k - P_i)
        RMSE_2 = math.sqrt(sum_2 / N_2)

        if RMSE_1 < RMSE_2 and RMSE_1 <= 0.9 * RMSE_2:
            self.mode = 1
        elif RMSE_2 < RMSE_1 and RMSE_2 <= 0.9 * RMSE_1:
            self.mode = -1
        else:
            self.states = 0
    def start(self):
        #对三个装甲板的旋转圆进行拟合
        for i in range(3):
            self.aims.append(fit_circle_fixed_radius(self.data[i],R=Const.R))
            # self.aims.append(self.fit_circle_3d(self.data[i]))
        #然后检查拟合效果
        self.check()
        if self.states==0:
            return
        array_center=np.array([self.aims[0][0],self.aims[1][0],self.aims[2][0]],dtype=np.float64)
        array_normal=np.array([self.aims[0][2],self.aims[1][2],self.aims[2][2]],dtype=np.float64)
        self.C = np.mean(array_center, axis=0)
        self.a = self.fit_parallel_direction(array_normal)
        for component in self.a[[1, 0, 2]]:  # 按 y、x、z 的优先级确定正方向
            if abs(component) > 1e-6:
                if component > 0:
                    self.a = -self.a
                break

        # 三个拟合圆心在旋转轴上的有符号坐标，由小到大对应 minus、middle、plus
        heights = (array_center - self.C) @ self.a
        self.minus_id, self.mid_id, self.plus_id = map(int, np.argsort(heights))
        self.update_e0_e90()

        # self.x_array=self.data_time[mid_id]
        # for point in self.data[mid_id]:
        #     theta=np.atan2(np.dot(point-self.C,self.e90),np.dot(point-self.C,self.e0))
        #     self.y_array.append(theta)
        # self.omega,self.theta0=self.fit_line(self.x_array,self.y_array)
        #   自己写的角速度拟合代码，atan2抖动太大,抢救多次无效,弃用...


        # """
        #GPT给的修正代码
        self.x_array = self.data_time[self.mid_id]
        points = np.asarray(self.data[self.mid_id], dtype=float)

        front_depth = self.C[2] - Const.R
        front_points = points * (front_depth / points[:, 2:3])
        cos_theta = ((front_points - self.C) @ self.e0) / Const.R

        front_sign = 1 if self.e90[2] < 0 else -1
        self.y_array = (front_sign * np.arccos(np.clip(cos_theta, -1, 1))).tolist()
        # """


        self.omega, self.theta0 = self.fit_line(self.x_array, self.y_array)
        
        #然后判断+-2/3pi
        #要求RMSE =sqrt(sum(||p_k - P_{i_k}(t_k)||²) / N) 先写一个函数求P_{i_k}(t_k)
        self.chose_model()
    def Get_plate_id(self,time_s,coordinate,rvec,tvec):
        #预测板号：
        #primary_plate_id 定义为本帧 used_for_state=1、质量检查通过且板号已知的候选中重投影误差最小者
        #取重投影误差最小值
        if not self.states:
            return "unknow"
        # aim_px=tools.projectPoints(coordinate,rvec,tvec)

        mid_pos=self.Predict_position("middle",time_s)
        plus_pos=self.Predict_position("plus",time_s)
        minus_pos=self.Predict_position("minus",time_s)

        # mid_px=tools.projectPoints(mid_pos,rvec,tvec)
        # plus_px=tools.projectPoints(plus_pos,rvec,tvec)
        # minus_px=tools.projectPoints(minus_pos,rvec,tvec)

        # mid_reprojection_error_px=tools.Get_reprojection_error([mid_px],[aim_px])
        # plus_reprojection_error_px=tools.Get_reprojection_error([plus_px],[aim_px])
        # minus_reprojection_error_px=tools.Get_reprojection_error([minus_px],[aim_px])

        # min_reprojection_error_px=min([mid_reprojection_error_px,plus_reprojection_error_px,minus_reprojection_error_px])
        # if min_reprojection_error_px==mid_reprojection_error_px:
        #     return "mid"
        # elif min_reprojection_error_px==plus_reprojection_error_px:
        #     return "plus"
        # elif min_reprojection_error_px==minus_reprojection_error_px:
        #     return "minus"

        mid_dist=np.dot(mid_pos-coordinate,mid_pos-coordinate)
        plus_dist=np.dot(plus_pos-coordinate,plus_pos-coordinate)
        minus_dist=np.dot(minus_pos-coordinate,minus_pos-coordinate)
        # print(mid_dist,minus_dist,plus_dist)
        min_dist=min([mid_dist,plus_dist,minus_dist])
        if min_dist==mid_dist:
            return "mid"
        elif min_dist==plus_dist:
            return "plus"
        elif min_dist==minus_dist:
            return "minus"
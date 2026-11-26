import numpy as np


def fit_circle_fixed_radius(data: list, R: float) -> tuple[np.ndarray, np.ndarray]:
    """以给定半径 R 拟合空间圆，使点到圆周的三维距离平方和尽量小。

    data 包含至少三个不共线的三维点，可按任意顺序输入。
    坐标和 R 均以 m 为单位。返回圆心和单位法向量，形状均为 (3,)。
    法向量绝对值最大的分量取正，用于统一正负号。
    """
    points = np.asarray(data, dtype=float)
    if points.ndim == 3 and points.shape[1:] in ((3, 1), (1, 3)):
        points = points.reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("data 至少需要 3 个三维点")
    R = float(R)
    if not np.isfinite(R) or R <= 0 or not np.isfinite(points).all():
        raise ValueError("坐标必须有限，R 必须是有限正数")

    # 以点集重心为原点，以 R 为长度单位，拟合圆的半径恒为 1。
    offsets = points - points[0]
    mean_offset = offsets.mean(axis=0)
    origin = points[0] + mean_offset
    q = (offsets - mean_offset) / R
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    if s[1] <= np.finfo(float).eps * max(q.shape) * s[0]:
        raise ValueError("点重合或共线，无法唯一确定圆平面")

    # 从初始平面估算圆心的方向和距重心的距离，只用于生成初始相位。
    xy = q @ vt[:2].T
    sq = np.sum(xy**2, axis=1)
    inward = np.linalg.lstsq(xy, sq - sq.mean(), rcond=None)[0]
    length = np.linalg.norm(inward)
    inward = inward / length if length > 0 else np.array([0.0, 1.0])
    across = xy @ inward
    along = xy @ np.array([-inward[1], inward[0]])
    # 对无噪声的单位圆，圆心到点集重心的距离平方等于 1 - sq.mean()。
    distance = np.sqrt(max(0.0, 1.0 - sq.mean()))

    best_cost = np.inf
    best = None
    # 从两个相反的圆心方向启动，保留收敛后误差较小的一组。
    for sign in (1.0, -1.0):
        theta = np.arctan2(along, distance - sign * across)
        last_cost = np.inf
        for _ in range(300):
            # cos(theta) - 1 的等价写法，在短弧的小角度下保留精度。
            template = np.column_stack((-2 * np.sin(theta / 2)**2, np.sin(theta)))
            template_mean = template.mean(axis=0)
            centered = template - template_mean

            # 固定相位时，SVD 求使三维距离平方和最小的圆平面基底。
            left, _, right_t = np.linalg.svd(q.T @ centered, full_matrices=False)
            basis = left @ right_t
            mean_radial = template_mean + np.array([1.0, 0.0])
            error = q - centered @ basis.T
            cost = np.sum(error**2)

            # 固定圆时，每个点取圆周上离它最近的相位。
            in_plane = q @ basis + mean_radial
            next_theta = np.arctan2(in_plane[:, 1], in_plane[:, 0])
            delta = next_theta - theta
            phase_step = np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
            if phase_step < 1e-12 or abs(last_cost - cost) <= 1e-13 * cost:
                break
            theta = next_theta
            last_cost = cost
        else:
            continue

        if cost < best_cost:
            best_cost = cost
            center = origin - R * (basis @ mean_radial)
            normal = np.cross(basis[:, 0], basis[:, 1])
            normal /= np.linalg.norm(normal)
            if normal[np.argmax(np.abs(normal))] < 0:
                normal = -normal
            best = center,R, normal

    if best is None:
        raise RuntimeError("圆拟合未收敛")
    return best

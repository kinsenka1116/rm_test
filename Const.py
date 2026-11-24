import numpy as np
image_width=1280
image_height=1024
w = 0.135
h = 0.056
objectPoints=np.array([
    [-w / 2, -h / 2, 0],
    [ w / 2, -h / 2, 0], 
    [ w / 2,  h / 2, 0],
    [-w / 2,  h / 2, 0]
],dtype=np.float64)
K=np.array([[1781.053454, 0, 510.5751561],[0, 1781.686873, 472.9735763],[0, 0, 1]],dtype=np.float64)
D=np.array([-0.04420740074, -0.1081537096,-0.0006506219039, 0.0002357380215, 0.466737247],dtype=np.float64)
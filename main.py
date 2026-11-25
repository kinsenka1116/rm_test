import cv2 as cv
import PnP.image_process as PnP
import input
import output
import tools
import fitting
import Const
data=[[],[],[]]#对应三层装甲板
data_time=[[],[],[]]
initializing=1
pointer=0
out=None
last_center=None
segment_id=0
None_aim_frame=0
fps=0
model=None
def frame_process(frame_id,time_s,frame):
    #先判断PnP状态，然后决定是否开始拟合
    #1.判断PnP状态
    global data,initializing,last_center,segment_id,pointer,None_aim_frame,fps,model,data_time,out
    img=frame.copy()
    result_PnP=PnP.image_process(img)
    #return aim_points,center,tvec[:,0],reprojection_error
    if isinstance(result_PnP,tuple):
        #PnP成功
        #然后判断和上一个有记录的帧是不是同一个装甲板
        #判断规则: 两者中心点的物理距离大于2块装甲板的宽度
        None_aim_frame=0
        coordinate=result_PnP[2]
        if last_center is not None:
            #等价于initializing=1
            if tools.Same_aim(coordinate,last_center):
                #未切换装甲板
                data_time[pointer].append(time_s)
                data[pointer].append(coordinate)
                last_center=coordinate
            else:
                #切换装甲板
                #此时可能可以开始拟合
                print("切换装甲板")
                cv.drawContours(frame, [result_PnP[0]], -1, (0, 255, 0), 2)
                if tools.check_fitting(data,fps):
                    #开始拟合
                    print("------------")
                    print(len(data[0]),len(data[1]),len(data[2]))
                    print(len(data_time[0]),len(data_time[1]),len(data_time[2]))
                    model=fitting.model(data,data_time)
                    model.start()
                    print(model.omega)
                    # exit(0)
                    pass
                last_center=coordinate
                pointer=(pointer+1)%3
                data_time[pointer]=[]
                data[pointer]=[]
                data_time[pointer].append(time_s)
                data[pointer].append(coordinate)
        else :
            #初始化
            segment_id+=1
            pointer=0
            data_time=[[],[],[]]
            data=[[],[],[]]
            last_center=coordinate
            data_time[pointer].append(time_s)
            data[pointer].append(coordinate)
    else :
        #PnP失败，记录reason
        #如果长时间PnP失败将重新初始化，令last_center=None
        None_aim_frame+=1
        if None_aim_frame/fps>0.5:
            #没有目标的持续时间>0.5s
            model=None#重新拟合
            last_center=None
    out.write(frame)
#frame_id,time_s,candidate_id=0,[aim_points],[center_u_px,center_v_px],plate_id,pnp_valid,tvec_x_m,tvec_y_m,tvec_z_m,reprojection_error_px,used_for_state,used_for_distance_filter,distance_raw_m,distance_filtered_m,reason
def video_process(video_path):
    global fps,out
    cap=cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(
    "./output/Output_video.avi",
    fourcc,
    fps,
    (Const.image_width, Const.image_height)
    )
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_id=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_s=cap.get(cv.CAP_PROP_POS_MSEC)/1000.0
        frame_process(frame_id,time_s,frame)
        # out.write(frame)
        frame_id+=1
    cap.release()
    out.release()
    cv.destroyAllWindows()
def main():
    output.Create_csv_observations()
    video_process("./input/video.avi")
if __name__=="__main__":
    main()
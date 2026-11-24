import cv2 as cv
import PnP.image_process as PnP
import input
import output
import Const
def frame_process(frame_id,time_s,frame):
    pass
#frame_id,time_s,candidate_id=0,[aim_points],[center_u_px,center_v_px],plate_id,pnp_valid,tvec_x_m,tvec_y_m,tvec_z_m,reprojection_error_px,used_for_state,used_for_distance_filter,distance_raw_m,distance_filtered_m,reason
def video_process(video_path):
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
        out.write(frame)
        frame_id+=1
    cap.release()
    out.release()
    cv.destroyAllWindows()
def main():
    output.Create_csv_observations()
    video_process("./input/video_first_2_seconds.avi")
if __name__=="__main__":
    main()
import csv
headers = [
    "frame_id",
    "time_s",
    "candidate_id",
    "corner0_u_px",
    "corner0_v_px",
    "corner1_u_px",
    "corner1_v_px",
    "corner2_u_px",
    "corner2_v_px",
    "corner3_u_px",
    "corner3_v_px",
    "center_u_px",
    "center_v_px",
    "plate_id",
    "pnp_valid",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "reprojection_error_px",
    "used_for_state",
    "used_for_distance_filter",
    "distance_raw_m",
    "distance_filtered_m",
    "reason"
]
def Create_csv_observations():
    with open("./output/data.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
#frame_id,time_s,candidate_id=0,[aim_points],[center_u_px,center_v_px],plate_id,pnp_valid,tvec_x_m,tvec_y_m,tvec_z_m,reprojection_error_px,used_for_state,used_for_distance_filter,distance_raw_m,distance_filtered_m,reason
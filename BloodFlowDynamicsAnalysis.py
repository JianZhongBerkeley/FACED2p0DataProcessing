import sys
import h5py
import numpy as np
from Packages.SimpleNeuronAnalysis.BloodFlowDynamics.FlowSpeedTraceOps import(
    stats_outlier_removal,
    win_stats_outlier_removal,
    trace_kalman_filter,
    eva_trace,
)

def analyze_blood_flow_dynamics(
    src_file_path,
    process_idx,
    stats_rm_win_size,
    stats_rm_std_ratio
):
    sys.dont_write_bytecode = True

    hybridvel_roi_nums = None
    hybridvel_angles = None

    with h5py.File(src_file_path, "r") as hdf5_file:
        hybridvel_roi_nums = hdf5_file["/roi_nums"][()]
        hybridvel_angles = hdf5_file["/angles"][()]

    hybridvel_roi_nums = np.rollaxis(hybridvel_roi_nums, -1, 0)
    hybridvel_roi_nums = np.squeeze(hybridvel_roi_nums)
    hybridvel_angles = np.rollaxis(hybridvel_angles, -1, 0)

    roi_sorted_order = np.argsort(hybridvel_roi_nums)

    hybridvel_roi_nums = hybridvel_roi_nums[roi_sorted_order]
    hybridvel_angles = hybridvel_angles[roi_sorted_order,...]

    process_speed_arr = hybridvel_angles[process_idx, :, -1, :].copy()
    nof_cands = process_speed_arr.shape[0]

    eva_res = np.zeros((nof_cands,),)
    process_clearned_speed_arr = np.zeros(process_speed_arr.shape)

    for i_cand in range(nof_cands):
        cur_plot_ys = process_speed_arr[i_cand, :]
        
        cur_plot_ys = stats_outlier_removal(cur_plot_ys)
        cur_stats_filterd_ys = win_stats_outlier_removal(cur_plot_ys, 
                                                        win_size = stats_rm_win_size, 
                                                        std_ratio = stats_rm_std_ratio)

        cur_eva_val = eva_trace(cur_stats_filterd_ys)

        eva_res[i_cand] = cur_eva_val
        process_clearned_speed_arr[i_cand, :] = cur_stats_filterd_ys

    selected_cand = np.argmax(eva_res)

    process_selected_speed = process_clearned_speed_arr[selected_cand, :]
    process_filtered_speed = trace_kalman_filter(process_selected_speed)
    
    return process_filtered_speed


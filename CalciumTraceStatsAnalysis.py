import sys
import tifffile
import numpy as np

from Packages.SimpleNeuronAnalysis.NeuralActivities.CalciumTraceOps import (
    suite2p_F_calculation,
    suite2p_F_neuropil_subtraction,
)
from Packages.SimpleNeuronAnalysis.NeuralActivities.CalciumTraceStats import (
    extract_roi_trace_group,
    calculate_response_group,
    stim_step_t_test,
    stim_step_anova_oneway,
    calculate_responses
)
from Packages.SimpleNeuronAnalysis.NeuralActivities.TimeStampGen import (
    vs_time_stamp_gen,
)
from Packages.SimpleNeuronAnalysis.OrientationTuning.OTAnalysis import (
    FitResponse,
    OSIndex,
    FittingUtils,
)


def calcium_trace_stats_analysis(
    src_data_path,
    src_mask_path,
    slice_num,
    init_acc_time_ms,
    time_per_slice_ms,
    time_per_vol_ms,
    static_moving_t_s,
    nof_orient,
    ops,
    i_roi,
    t_test_alpha,
    anova_test_alpha,
):
    sys.dont_write_bytecode = True

    s_to_ms = 1e3

    _, stim_tstamp_s, _, orient_angles_rad = vs_time_stamp_gen(
        static_moving_t_s,
        nof_orient,
        time_per_vol_ms,
    )
    response_stim_offset = int(np.round(1E3/time_per_vol_ms))
    stim_start_offset_ms = - (init_acc_time_ms + slice_num * time_per_slice_ms)
    stim_tstamp_shifted_ms = stim_tstamp_s * s_to_ms  + stim_start_offset_ms
    stim_tstamp_shifted = stim_tstamp_shifted_ms / time_per_vol_ms

    src_mask = tifffile.imread(src_mask_path)

    roi_idx_to_label_offset = 1

    if len(src_mask.shape) == 2:
        src_mask = src_mask[np.newaxis,...]

    nof_rois = src_mask.shape[0]
    mask_height =  src_mask.shape[1]
    mask_width =  src_mask.shape[2]

    src_labeled_mask = np.zeros((mask_height, mask_width))
    for i_roi in range(nof_rois):
        src_labeled_mask[src_mask[i_roi,:,:] > 0] = i_roi + roi_idx_to_label_offset

    assert(np.unique(src_labeled_mask)[roi_idx_to_label_offset:].size == nof_rois)

    mask_height = src_labeled_mask.shape[0]
    mask_width = src_labeled_mask.shape[1]

    src_data = tifffile.imread(src_data_path)  
    nof_traces, trace_len, _, _ = src_data.shape

    Fs = np.zeros((nof_rois, nof_traces, trace_len))
    Fneus = np.zeros((nof_rois, nof_traces, trace_len))

    for i_trace in range(nof_traces):
        F, Fneu = suite2p_F_calculation(src_data[i_trace,:,:,:], src_labeled_mask, ops)
        Fs[:,i_trace,:] = F
        Fneus[:,i_trace,:] = Fneu

    blank_len = int(np.round(1E3/time_per_vol_ms))
    stim_append_len = int(static_moving_t_s[-1]*s_to_ms/time_per_vol_ms)
    nof_orints = stim_tstamp_shifted.shape[0]
    blank_tstamps = np.zeros((nof_orints, 2), dtype = int)
    blank_tstamps[:,1] = stim_tstamp_shifted[:,0,1]
    blank_tstamps[:,0] = stim_tstamp_shifted[:,0,1] - blank_len
    stim_tstamps = np.zeros((nof_orints, 2), dtype = int)
    stim_tstamp_len = int(np.min(stim_tstamp_shifted[:,1,1] - stim_tstamp_shifted[:,1,0]))
    stim_tstamps[:,0] = stim_tstamp_shifted[:,1,0]
    stim_tstamps[:,1] = stim_tstamp_shifted[:,1,0] + stim_tstamp_len + stim_append_len

    stim_Fs_group = extract_roi_trace_group(Fs, stim_tstamps)
    blank_Fs_group = extract_roi_trace_group(Fs, blank_tstamps)
    stim_Fneus_group = extract_roi_trace_group(Fneus, stim_tstamps)
    blank_Fneus_group = extract_roi_trace_group(Fneus, blank_tstamps)
    Fneus_group_F0 = np.mean(blank_Fneus_group, axis = (2,3))
    Fneus_group_F0 = Fneus_group_F0.reshape((Fneus_group_F0.shape[0], Fneus_group_F0.shape[1], 1, 1))
    stim_FmFneu_group = suite2p_F_neuropil_subtraction(stim_Fs_group, stim_Fneus_group - Fneus_group_F0, ops)
    blank_FmFneu_group = suite2p_F_neuropil_subtraction(blank_Fs_group, blank_Fneus_group - Fneus_group_F0, ops)
    FmFneu_blank_AvgF = np.mean(blank_FmFneu_group, axis = (2,3))
    FmFneu_group_F0 = FmFneu_blank_AvgF
    FmFneu_group_F0 = FmFneu_group_F0.reshape((FmFneu_group_F0.shape[0], FmFneu_group_F0.shape[1], 1, 1))
    stim_FmFnue_group_dFF = (stim_FmFneu_group - FmFneu_group_F0)/FmFneu_group_F0
    blank_FmFnue_group_dFF = (blank_FmFneu_group - FmFneu_group_F0)/FmFneu_group_F0

    cur_stim_dFF = stim_FmFnue_group_dFF[i_roi, :, :]
    cur_blank_dFF = blank_FmFnue_group_dFF[i_roi, :, :]
    response_group = calculate_response_group(cur_blank_dFF, cur_stim_dFF[:,:,response_stim_offset:])
    nof_orints = response_group.shape[0]
    nof_traces = response_group.shape[2]
    t_test_pvals, _ = stim_step_t_test(response_group, test_steps = [1,0])
    anova_test_result = stim_step_anova_oneway(response_group, test_step = 1)
    t_test_effective_alpha = t_test_alpha/nof_orints
    t_test_pass = np.min(t_test_pvals) <= t_test_effective_alpha
    anova_test_pass = anova_test_result.pvalue <= anova_test_alpha
    responses = calculate_responses(response_group)

    roi_responses = responses
    fit_bounds = FittingUtils.est_double_gauss_fit_bounds(orient_angles_rad, roi_responses)
    double_gaussian_fit_obj = FitResponse.DoubleGaussian()
    double_gaussian_fit_obj.fit(orient_angles_rad, 
                                roi_responses,
                                bounds = fit_bounds)
    roi_OSI = OSIndex.calculate_OSI(double_gaussian_fit_obj)
    roi_DSI = OSIndex.calculate_DSI(double_gaussian_fit_obj)

    return (t_test_pass, anova_test_pass, roi_OSI, roi_DSI, double_gaussian_fit_obj)


# run this script as demo
if __name__ == "__main__":
    print("Usage:")
    print("* Before using this script, perform motion registartion on the volumetric data set and split the volumetric imaging data by Z slices")
    print("* Set src_data_path as the file path for file path for the image sequence at one Z slice.")
    print("* Set src_mask_path as the file path for the ROIs (ROIs should be saved as a mask image stack)")
    print("* Provide experiment configurations for rest of the parameters according to the parameter name")
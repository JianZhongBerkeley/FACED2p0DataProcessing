import sys

import numpy as np
import h5py
import os

from Packages.SimpleNeuronAnalysis.NeuralActivities.VoltageTraceOps import (
    moving_avg,
)
from Packages.SimpleNeuronAnalysis.NeuralActivities.PairwiseCorrelation import (
    norm_cov_matrix,
    shift_correct_cov_matrix,
)

def voltage_imaging_pairwise_cc(
    time_per_frame_ms,
    src_file_path,
    win_size,
    pre_rm_size,
):    
    sys.dont_write_bytecode = True

    ms_to_s = 1e-3

    nof_rois = None
    nof_trials = None
    nof_frames = None

    roi_spike_events = None
    roi_subthd_dFFs = None

    with h5py.File(src_file_path, "r") as hdf5_file:
        nof_rois = hdf5_file["nof_roi"][()]
        nof_frames = hdf5_file["nof_frames"][()]
        nof_trials = hdf5_file["nof_files"][()]

        roi_spike_events = np.zeros((nof_rois, nof_trials, nof_frames))
        roi_subthd_dFFs = np.zeros((nof_rois, nof_trials, nof_frames))
        
        for i_roi in range(nof_rois):
            cur_roi_str = f"roi{i_roi}"
            roi_spike_events[i_roi, :, :] = hdf5_file[os.path.join(cur_roi_str, "cur_spike_event")][()]
            roi_subthd_dFFs[i_roi, :, :] = hdf5_file[os.path.join(cur_roi_str, "cur_subthreshold_dFF")][()]

    FR_dFF_win_size = int(np.ceil(win_size/time_per_frame_ms))
    pre_rm_nof_frames = int(np.ceil(pre_rm_size/time_per_frame_ms))

    input_subthd_dFF = roi_subthd_dFFs[:,:,pre_rm_nof_frames:]
    input_spike_events = roi_spike_events[:,:,pre_rm_nof_frames:]

    nof_rois, nof_trials, _ = input_subthd_dFF.shape

    FRdFF_FRs = moving_avg((input_spike_events > 0).astype(float), FR_dFF_win_size)
    FRdFF_dFFs = moving_avg(input_subthd_dFF.astype(float), FR_dFF_win_size)

    FRdFF_FRs = FRdFF_FRs/(time_per_frame_ms * ms_to_s)

    input_suprathd_FRs = FRdFF_FRs
    input_subthd_dFFs = FRdFF_dFFs

    suprathd_shift_cov_matrix = shift_correct_cov_matrix(input_suprathd_FRs)
    subthd_shift_cov_matrix = shift_correct_cov_matrix(input_subthd_dFFs)

    suprathd_shift_cc_matrix = norm_cov_matrix(suprathd_shift_cov_matrix)
    subthd_shift_cc_matrix = norm_cov_matrix(subthd_shift_cov_matrix)

    input_suprathd_matrix = suprathd_shift_cc_matrix
    input_subthd_matrx = subthd_shift_cc_matrix

    cur_nof_rois = input_subthd_matrx.shape[0]
    cur_indices = np.triu_indices(cur_nof_rois, k = 1)

    xs = input_subthd_matrx[cur_indices]
    ys = input_suprathd_matrix[cur_indices]

    return (xs, ys)




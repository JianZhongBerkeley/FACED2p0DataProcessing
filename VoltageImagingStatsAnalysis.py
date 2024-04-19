import sys
import numpy as np
import h5py
import os
import re

from Packages.SimpleNeuronAnalysis.NeuralActivities.VoltageTraceStats import (
    cnt_spike_in_time_stamp,
    calculate_spike_rate_s,
    stim_step_t_test,
    stim_step_anova_oneway,
    calculate_suprathd_responses,
    calculate_subthd_responses,
    holm_bonferrioni_comparison,
)
from Packages.SimpleNeuronAnalysis.NeuralActivities.VoltageTraceOps import (
    bw_lp_filtering,
)
from Packages.SimpleNeuronAnalysis.NeuralActivities.TimeStampGen import (
    vs_time_stamp_gen,
)
from Packages.SimpleNeuronAnalysis.IO.FcddatPkgOps import (
    listPkgNames,
    sortPkgNames,
)
from Packages.SimpleNeuronAnalysis.OrientationTuning.OTAnalysis import (
    FitResponse,
    OSIndex,
    FittingUtils,
)


def voltage_imaging_stats_analysis(
    src_root_dir,
    src_sub_dir_name,
    src_hdf5_file_name,
    nframes_key,
    dFF_key,
    roi_regex,
    fields,
    i_roi,
    process_field,
    time_per_frame_ms,
    bw_order,
    bw_cutoff,
    static_moving_t_s,
    nof_orient,
    t_test_alpha,
    anova_test_alpha,
):
    
    ms_to_s = 1e-3
    s_to_ms = 1e3

    sys.dont_write_bytecode = True

    fs = 1 * s_to_ms / time_per_frame_ms

    stim_tstamp, stim_tstamp_s, _, orient_angles_rad = vs_time_stamp_gen(
        static_moving_t_s,
        nof_orient,
        time_per_frame_ms,
    )
    stim_tstamp_s = 0.25 + stim_tstamp_s
    stim_tstamp_s[stim_tstamp_s > (3100-1)*time_per_frame_ms*ms_to_s] = (3100-1)*time_per_frame_ms*ms_to_s

    stim_tstamp = (stim_tstamp_s * s_to_ms)/time_per_frame_ms
    stim_tstamp = stim_tstamp.astype(int)

    src_pkg_names = listPkgNames(src_root_dir)
    src_pkg_names = sortPkgNames(src_pkg_names)

    nof_fields = len(fields)
    nof_roi = 0
    nof_files = len(src_pkg_names)
    nof_frames = 0

    spike_events = None

    for ipkg_name in [src_pkg_names[0]]:
        src_hdf5_file_path = os.path.join(src_root_dir, ipkg_name, src_sub_dir_name, src_hdf5_file_name)
        with h5py.File(src_hdf5_file_path, "r") as hdf5_file:
            nof_frames = hdf5_file[nframes_key][()].shape[0]
            max_roi_num = 0
            for key in hdf5_file.keys():
                for istr in re.findall(roi_regex, key):
                    cur_roi_num = int(istr[3:])
                    max_roi_num = max(max_roi_num, cur_roi_num)
            nof_roi = max_roi_num + 1
            
    spike_events = np.zeros((nof_roi, nof_fields, nof_files, nof_frames))
    dFFs = np.zeros((nof_roi, nof_files, nof_frames))

    for ifile in range(nof_files):
        src_hdf5_file_path = os.path.join(src_root_dir, src_pkg_names[ifile], src_sub_dir_name, src_hdf5_file_name)
        with h5py.File(src_hdf5_file_path, "r") as hdf5_file:
            for iroi in range(nof_roi):
                for ifield in range(nof_fields):
                    spike_idxs = hdf5_file[fields[ifield].format(iroi = iroi)][()]
                    spike_events[iroi, ifield, ifile, spike_idxs] = 1
                dFFs[iroi, ifile, :] = hdf5_file[dFF_key.format(iroi = iroi)][()]


    cur_dFF = dFFs[i_roi, :, :]
    cur_spike_event = spike_events[i_roi, process_field, :, :]

    cur_subthreshold_dFF = bw_lp_filtering(
        order = bw_order,
        cutoff = bw_cutoff, 
        fs = fs, 
        src_traces = cur_dFF,
    )

    cur_spike_cnts = cnt_spike_in_time_stamp(cur_spike_event, stim_tstamp)
    cur_spike_rate = calculate_spike_rate_s(cur_spike_cnts, stim_tstamp_s)

    t_test_pvals, _ = stim_step_t_test(cur_spike_rate, test_steps = [1,0])
    anova_test_result = stim_step_anova_oneway(cur_spike_rate, test_step = 1)

    t_test_pass = holm_bonferrioni_comparison(t_test_pvals, t_test_alpha)
    anova_test_pass = anova_test_result.pvalue < anova_test_alpha

    suprathd_responses = calculate_suprathd_responses(cur_spike_rate)
    subthd_responses = calculate_subthd_responses(cur_subthreshold_dFF, stim_tstamp)

    roi_fr_responses = suprathd_responses

    fr_bounds = FittingUtils.est_double_gauss_fit_bounds(orient_angles_rad, roi_fr_responses)

    fr_double_gaussian_fit_obj = FitResponse.DoubleGaussian()
    fr_double_gaussian_fit_obj.fit(orient_angles_rad, 
                                roi_fr_responses,
                                bounds = fr_bounds)

    fr_OSI = OSIndex.calculate_OSI(fr_double_gaussian_fit_obj)

    roi_subthd_responses = subthd_responses

    subthd_bounds = FittingUtils.est_double_gauss_fit_bounds(orient_angles_rad, roi_subthd_responses)

    subthd_double_gaussian_fit_obj = FitResponse.DoubleGaussian()
    subthd_double_gaussian_fit_obj.fit(orient_angles_rad, 
                                    roi_subthd_responses,
                                    bounds = subthd_bounds)

    subthd_OSI = OSIndex.calculate_OSI(subthd_double_gaussian_fit_obj)

    return (t_test_pass, anova_test_pass, fr_OSI, subthd_OSI, fr_double_gaussian_fit_obj, subthd_double_gaussian_fit_obj)

    




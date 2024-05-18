import sys

import os 
import glob
import numpy as np

from Packages.SimpleNeuronAnalysis.VolClustering.ROIStackOps import (
    AddDataClusterProperty,
    CleanROIChain,
    CreateDataCluster,
    CreateROIChain,
    VerifyROIStack,
    SelectRepresentativeROI,
)


def volumetric_imaging_cell_clustering(
    src_data_root_dir_path,
    src_data_dir_name_glob,
    src_data_file_name_glob,
    image_ij_pixelsize_um,
    image_z_stepsize_um,
    src_slice_num_regex,
    neuropil_subtract_rate,
    min_feature_pcc,
    max_roi_distance_um,
    snr_est_lp_sigma
):
    
    sys.dont_write_bytecode = True

    src_data_file_path_glob = os.path.join(src_data_root_dir_path, src_data_dir_name_glob, src_data_file_name_glob)
    src_data_file_paths = glob.glob(src_data_file_path_glob)
    src_data_file_paths = np.array(src_data_file_paths)
    nof_data_files = len(src_data_file_paths)

    src_data_file_names = [None for _ in range(nof_data_files)]
    for i_file in range(nof_data_files):
        src_data_file_names[i_file] = os.path.split(src_data_file_paths[i_file])[-1]
    src_data_file_names = np.array(src_data_file_names)

    file_name_sorted_idxs = np.argsort(src_data_file_names)
    src_data_file_names = src_data_file_names[file_name_sorted_idxs]
    src_data_file_paths = src_data_file_paths[file_name_sorted_idxs]

    src_data_cluster = CreateDataCluster.create_data_cluster_from_hdf5(src_data_file_paths, src_slice_num_regex)
    print(f"ascending slice order: {VerifyROIStack.verify_ascending_order(src_data_cluster)}")
    AddDataClusterProperty.add_roi_center_ijs(src_data_cluster)
    AddDataClusterProperty.add_roi_center_yxz_ums(src_data_cluster, image_ij_pixelsize_um, image_z_stepsize_um)
    AddDataClusterProperty.add_mix_trial_avg_dFF(src_data_cluster)
    AddDataClusterProperty.add_continous_dFF(src_data_cluster, neuropil_subtract_rate = neuropil_subtract_rate)
    AddDataClusterProperty.add_roi_feature(src_data_cluster, "FmFneu_continous_dFFs")
    CreateROIChain.create_roi_chain(src_data_cluster)
    CleanROIChain.remove_impossible_links(src_data_cluster, max_roi_distance_um, min_feature_pcc)
    CleanROIChain.clustering_roi_chain(src_data_cluster, max_roi_distance_um, min_feature_pcc)

    all_slice_idxs, all_roi_idxs = SelectRepresentativeROI.get_representative_all_rois(src_data_cluster, snr_est_lp_sigma)
    ve_slice_idxs, ve_roi_idxs = SelectRepresentativeROI.get_representative_ve_rois(src_data_cluster, snr_est_lp_sigma)
    os_slice_idxs, os_roi_idxs = SelectRepresentativeROI.get_representative_os_rois(src_data_cluster, snr_est_lp_sigma)

    return (all_slice_idxs, all_roi_idxs, ve_slice_idxs, ve_roi_idxs, os_slice_idxs, os_roi_idxs)


# run this script as demo
if __name__ == "__main__":
    all_slice_idxs, _, ve_slice_idxs, _, _, _ = volumetric_imaging_cell_clustering(
        "./DemoData/Calcium/demo_data",
        "slice" + "[0-9]" * 2,
        "slice" + "[0-9]" * 2 + ".hdf5",
        (2, 1.389),
        10,
        r"slice\d+",
        0.7,
        0.5,
        25,
        1.635,
    )
    print(f"Total nof cells: {len(all_slice_idxs)}")
    print(f"Total nof VE cells: {len(ve_slice_idxs)}")


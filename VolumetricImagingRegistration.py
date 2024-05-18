# NOTE: The image registration script was adopted from the example script from CaImAn
# REFERENCE: Kleinfeld, D. et al. CaImAn an open source tool for scalable calcium imaging data analysis. (2019) doi:10.7554/eLife.38173.001.

import sys

import os
from tifffile.tifffile import imwrite
import glob

import caiman as cm

import cv2
import numpy as np
import os.path
import logging

import pickle

import caiman as cm
from caiman.motion_correction import MotionCorrect
import scipy.ndimage as ndimage

from Packages.SimpleNeuronAnalysis.IO.FcddatPkgOps import (
    listPkgNames,
    getPkgFNum,
)


def volumetric_imaging_registration(
    src_root_dir_path,
    process_slice_idx,
    src_slice_sub_dir_name,
    src_slice_file_name_glob,
    dst_root_dir_path,
    dst_dir_name_pattern,
    max_shifts,
    strides,
    overlaps,
    max_deviation_rigid,
    pw_rigid,
    shifts_opencv,
    border_nan,
    min_mov,
):
    sys.dont_write_bytecode = True

    try:
        cv2.setNumThreads(0)
    except:
        pass

    try:
        if __IPYTHON__:
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        pass

    logging.basicConfig(format=
                            "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        level=logging.DEBUG)
    

    src_pkg_names = listPkgNames(src_root_dir_path)
    assert(len(src_pkg_names) > 0)    
    src_pkg_names.sort(key = getPkgFNum)

    search_pkg_name = src_pkg_names[0]
    search_slice_file_path_glob = os.path.join(
        src_root_dir_path,
        search_pkg_name,
        src_slice_sub_dir_name,
        src_slice_file_name_glob
        )
    search_slice_file_path_result = glob.glob(search_slice_file_path_glob)
    src_slice_file_names = [os.path.split(file_path)[-1] for file_path in search_slice_file_path_result]
    src_slice_file_names.sort(reverse = True)

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    _, dview, _ = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    
    nof_trails = min(nof_trails, len(src_pkg_names))

    dst_dir_path = os.path.join(dst_root_dir_path, 
                                dst_dir_name_pattern.format(nof_trails = nof_trails))
    if not os.path.isdir(dst_dir_path):
        os.makedirs(dst_dir_path)

    dst_mcobj_pickel_file_path = os.path.join(dst_dir_path, "reg_mc_objs.pkl")
    if os.path.exists(dst_mcobj_pickel_file_path):
        with open(dst_mcobj_pickel_file_path, "rb") as pkl_file:   
            mcobj_dict = pickle.load(pkl_file)
            mc = mcobj_dict.get("mc", None)
            mc_pwrigid = mcobj_dict.get("mc_pwrigid", None)
        
    src_slice_file_name = src_slice_file_names[process_slice_idx]

    fnames = []
    for i_pkg in range(nof_trails):
        src_pkg_name = src_pkg_names[i_pkg]
        src_file_path = os.path.join(src_root_dir_path, 
                                    src_pkg_name, 
                                    src_slice_sub_dir_name,
                                    src_slice_file_name)
        if not os.path.exists(src_file_path):
            break
        fnames.append(src_file_path)
        
    if len(fnames) < nof_trails:
        print("nof files < nof trials")
        
    mc = None
    if mc is None:
        mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                        strides=strides, overlaps=overlaps,
                        max_deviation_rigid=max_deviation_rigid, 
                        shifts_opencv=shifts_opencv, nonneg_movie=True,
                        border_nan=border_nan, min_mov = min_mov)
        mc.pw_rigid = False
        mc.motion_correct(save_movie=True)
        
    mc_pwrigid = None
    if pw_rigid and mc_pwrigid is None:
        low_pass_mv = cm.load(mc.mmap_file)
            
        nof_frames = np.int(low_pass_mv.shape[0]/nof_trails)
                    
        low_pass_mv = ndimage.gaussian_filter1d(low_pass_mv, sigma = nof_frames/16, truncate = 2, axis = 0)
        ref_tif_file_path = os.path.join(dst_dir_path, "ref_lowpass.tif")
            
        imwrite(ref_tif_file_path, low_pass_mv)        
        mc_pwrigid = MotionCorrect(ref_tif_file_path, dview=dview, max_shifts=max_shifts,
                                    strides=strides, overlaps=overlaps,
                                    max_deviation_rigid=max_deviation_rigid, 
                                    shifts_opencv=shifts_opencv, nonneg_movie=True,
                                    border_nan=border_nan, min_mov = min_mov,
                                    niter_rig = 1)
        mc_pwrigid.pw_rigid = False
        mc_pwrigid.motion_correct(save_movie=True)
        mc_pwrigid.pw_rigid = True
        mc_pwrigid.template = mc_pwrigid.mmap_file
        mc_pwrigid.motion_correct(save_movie=True, template=mc_pwrigid.total_template_rig)
        ref_reg_tif_path = os.path.join(dst_dir_path, "ref_reg.tif")
        imwrite(ref_reg_tif_path, cm.load(mc_pwrigid.mmap_file))
        
        low_pass_mv = None
        
    with open(dst_mcobj_pickel_file_path, "wb") as pkl_file:
        mcobj_dict = {
            "mc": mc,
            "mc_pwrigid": mc_pwrigid,
        }
        pickle.dump(mcobj_dict, pkl_file, pickle.HIGHEST_PROTOCOL)

    cm.stop_server(dview=dview)


# run this script as 
if __name__ == "__main__":
    print("Usage:")
    print("* Before using this script, split the volumetric imaging data for each trial by Z slices and save them into a subdirectory inside the directory for each trail.")
    print("* Set src_root_dir_path as the path for the directory which includes all the directories for all trials.")
    print("* Set process_slice_idx to be index of the slice you would like process.")
    print("* Set src_slice_sub_dir_name to be the name of the sub directory which contains data for each Z slice")
    print("* Set src_slice_file_name_glob to be the name pattern on the files for data in each Z slice.")
    print("* Use dst_root_dir_path and dst_dir_name_pattern to configure the output directory.")
    print("* Set up other parameters required by CaImAn motion correction.")
    print("NOTE: Please check the following CaImAn notebook for demonstration on how to use the CaImAn image registration:")
    print("URL: https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_motion_correction.ipynb")

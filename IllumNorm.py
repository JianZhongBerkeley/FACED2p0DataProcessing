import sys
import skimage
import numpy as np

from Packages.SimpleNeuronAnalysis.IllumNorm.LineScanNormOps import (
    percentile_illum_profile,
    illum_norm_image,
    illum_est_loss,
)


# function to perform linescan illumination normalization of FACED line scan
def illum_norm(
    src_file_path,
):
    sys.dont_write_bytecode = True

    src_raw_image = skimage.io.imread(src_file_path)
    input_image = src_raw_image
    test_percentiles = np.linspace(0,100, 101)
    test_losses = np.zeros(test_percentiles.shape)

    for i_prct in range(len(test_percentiles)):
        cur_percentile = test_percentiles[i_prct]
        cur_loss = illum_est_loss(input_image, 
                                percentile_illum_profile, 
                                [1, cur_percentile, 200, "none"])
        
        test_losses[i_prct] = cur_loss

    best_est_idx = np.argmin(test_losses)
    best_est_percentile = test_percentiles[best_est_idx]

    dst_image = illum_norm_image(input_image, 
                                percentile_illum_profile, 
                                [1, best_est_percentile, 200, "none"])
    
    return dst_image


## run this script as demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    demo_src_file_path = "./DemoData/IllumNorm/demo_data.tif"
    demo_result = illum_norm(demo_src_file_path)

    plt.figure()
    plt.imshow(demo_result, cmap = "gray")
    plt.show()




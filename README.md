# FACED2p0DataProcessing
Data analysis code for the paper entitled *FACED 2.0 enables large-scale voltage and calcium imaging in vivo*

Detailed description of the functions can be found in the method parts of the above paper.

## Citation
Please cite the *FACED 2.0 enables large-scale voltage and calcium imaging in vivo* paper if you use the code in this repository.  

## Dependency
- numpy
- scipy
- h5py
- matplotlib
- skcikit-learn
- skcikit-image
- tiffile
- pykalman
- suite2p
- caiman

## System requirements 
Te run the code, we recommend to set up workstation with the following computation resources:
- CPU with over 3.0 GHz clock rate and 18 cores
- over 128 GB RAM
- SATA 3.0 SSD with over 2TB storage capacity
- GPU with over 4.0 GB memory
- OS: Windows 11 Pro

## Installation guide
1. Follow installation instructions on [Python](https://www.python.org/) webiste to install python compiler to your computer.
2. Follow installation instructions on [numpy](https://numpy.org/), [SciPy](https://scipy.org/), [H5py](https://www.h5py.org/), [matplotlib](https://matplotlib.org/), [skcikit-learn](https://scikit-learn.org/stable/), and [scikit-image](https://scikit-image.org/) websites to install these modules for your python compiler.
3. Follow installation instructions on [pykalman git repo](https://github.com/pykalman/pykalman) to install pykalman for your python compiler
4. Follow installation instructions on [suite2p git repo](https://github.com/MouseLand/suite2p) to install suite2p for your python compiler.
5. Follow installation instructions on [CaImAn git repo](https://github.com/flatironinstitute/CaImAn) to install CaImAn for your python compiler.
6. Git clone or git add submodule of this repo to your project and you are ready to go.
NOTE: Installing dependecy highly depends on the network environment and the dependency host server for these denedencies. It can vary from minutes to hours. Git cloning this repo should only takes less than a couple mintues.

## How to use the code
The python script under the top-level directory can be run by simply using the following command:
```
python "./script.py"
```
Demo script will be excuted and output will be reported.

You can also directly `import` the scripts into your project and use their functions for your own data processing. 


## License
[GLP-3.0 license](./LICENSE)

## Reference

- Kleinfeld, D. et al. CaImAn an open source tool for scalable calcium imaging data analysis. (2019) doi:10.7554/eLife.38173.001.
- Cai, C. et al. VolPy: Automated and scalable analysis pipelines for voltage imaging datasets. PLoS Comput Biol 17, (2021).
- Chhatbar, P. Y. & Kara, P. Improved blood velocity measurements with a hybrid image filtering and iterative Radon transform algorithm. Front Neurosci (2013) doi:10.3389/fnins.2013.00106.
- Pachitariu, M. et al. Suite2p: beyond 10,000 neurons with standard two-photon microscopy. doi:10.1101/061507.
- Duckworth, D. et al. pykalman. GitHub repository at https://github.com/pykalman/pykalman (2012).



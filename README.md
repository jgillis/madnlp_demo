To run the demo:
 * get access to a workstation with linux and a CUDA-enablde GPU
 * pip install CasADi 3.6.6 or a nightly build https://github.com/casadi/casadi/releases/tag/nightly-main if 3.6.6 is not released yet
 * obtain libmadnlp_c.so and make it findable by CasADi (e.g. with LD_LIBRARY_PATH env var) - from https://github.com/tmmsartor/madnlp_c/releases/download/nightly-gpu_lib/madnlp-jl1.10.4-ubuntu-20.04-x64.zip
 * `python demo.py`

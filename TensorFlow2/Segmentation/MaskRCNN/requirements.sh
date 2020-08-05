pip --no-cache-dir --no-cache install \
        Cython \
        matplotlib \
        opencv-python-headless \
        mpi4py \
        Pillow \
        pytest \
        pyyaml

# Install pybind11
git clone https://github.com/pybind/pybind11
cd pybind11
cmake .
sudo make -j96 install
pip install .

# Install NVIDIA COCO and dllogger
pip --no-cache-dir --no-cache install \
    'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI' && \
pip --no-cache-dir --no-cache install \
    'git+https://github.com/NVIDIA/dllogger'


pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
CUDA_HOME=/shared/tensorflow_env/ pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda100


WORK_DIR=`mktemp -d`

pip install ninja yacs cython matplotlib tqdm opencv-python pybind11==2.5.0 'git+https://github.com/NVIDIA/dllogger'

pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

cd $WORK_DIR
git clone https://github.com/NVIDIA/apex; cd apex;
python setup.py install --cuda_ext --cpp_ext

cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN/pytorch/
python setup.py build develop

rm -rf $WORK_DIR
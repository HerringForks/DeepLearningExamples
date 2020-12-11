pip install h5py boto3 'git+https://github.com/NVIDIA/dllogger' tqdm requests
WORK_DIR=`mktemp -d`
cd $WORK_DIR

git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cuda_ext --cpp_ext
rm -rf $WORK_DIR
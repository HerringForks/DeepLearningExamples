pip install Cython
pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'

pip install pycocotools==2.0.0 contextlib2
pip install Pillow
pip install matplotlib
pip install numpy==1.17.5 --upgrade

cd /shared/DeepLearningExamplesKaran/TensorFlow/Detection/SSD/models/research
protoc object_detection/protos/*.proto --python_out=.
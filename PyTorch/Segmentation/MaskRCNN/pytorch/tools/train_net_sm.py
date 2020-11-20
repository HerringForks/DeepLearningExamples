import argparse
import os
import subprocess
import time
import datetime
import shutil
import json


# def parse_args():
#     parser = argparse.ArgumentParser(description='Get model info')
#     parser.add_argument('--num_nodes', type=int, help='Number of nodes')
#     parser.add_argument('--model', type=str, help='One of mrcnn or bert')
#     parser.add_argument('--data_mode', type=str, help='One of fsx or s3')
#     args = parser.parse_args()
#     return args

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Override number of training steps in the config")
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model",
                        action="store_true", )
    parser.add_argument("--fp16", help="Mixed precision training", action="store_true")
    parser.add_argument("--amp", help="Mixed precision training", action="store_true")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument("--json-summary", help="Out file for DLLogger", default="dllogger.out",
                        type=str,
                        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        help="Absolute path of dataset ",
        type=str,
        default=None
    )
    parser.add_argument(
        "--seed",
        help="manually set random seed for torch",
        type=int,
        default=99
    )

    args, unknown = parser.parse_known_args()
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_net.py'))
    config_file = os.path.join('/opt/ml/code', args.config_file)
    num_nodes = len(os.environ["SM_HOSTS"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    work_dir = os.environ['SM_OUTPUT_DATA_DIR']

    data_dir = "/opt/ml/input/data/train/"
    json_file = work_dir + "/dllogger.json"

    # cmd = f"python -m torch.distributed.launch --nnodes={num_nodes}  --node_rank={rank}  --nproc_per_node={num_gpus} \
    #     --master_addr={hosts[0]} --master_port='1234' \
    #     /opt/ml/code/DeepLearningExamples/PyTorch/Segmentation/MaskRCNN/pytorch/tools/train_net.py --data-dir {data_dir} \
    #     --config-file /opt/ml/code/test_config.yaml --seed 987 --skip-test DTYPE float16 OUTPUT_DIR {work_dir}"


    cmd = (
        f'python'
        f' -m torch.distributed.launch'
        f' --nnodes={num_nodes}'
        f' --node_rank={rank}'
        f' --nproc_per_node={num_gpus}'
        f' --master_addr={hosts[0]}'
        f' --master_port="12345"'
        f' {main_path}'
        f' --data-dir {data_dir}'
        f' --config-file {config_file}'
        f' --seed {args.seed}'
        f' --skip-test'
        f' DTYPE float16'
        f' OUTPUT_DIR {work_dir}'
    )

    start_timestamp = datetime.datetime.now()
    start_timestamp = start_timestamp.strftime("%Y-%m-%d-%H-%M-%S")

    start = time.time()
    invoke_train(cmd)
    end = time.time()

def invoke_train(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def export_to_s3(work_dir, model_dir, totaltime, model_name, timestamp):
    print("################################")
    print(f"Time taken is: {totaltime}")
    print("################################")

    time_file = os.path.join(work_dir, "timetaken")

    with open(time_file, 'w') as file:
        file.write(str(totaltime))

    dst = os.path.join(model_dir, "timetaken")
    shutil.copyfile(time_file, dst)


if __name__ == "__main__":

    # args = parse_args()
    # num_nodes = args.num_nodes
    # model = args.model
    # data_mode = args.data_mode

    main()
    # export_to_s3(work_dir, os.environ['SM_MODEL_DIR'], end-start, args.model_name, start_timestamp)

import argparse
import os
import subprocess
import sys
import io
import time
import datetime
import shutil
import json

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def parse_args():
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes')
    parser.add_argument('--train_batch_size', type=int, help='train_batch_size')
    parser.add_argument('--max_seq_length', type=int, help='max_seq_length')
    parser.add_argument('--max_predictions_per_seq', type=int, help='max_predictions_per_seq')
    parser.add_argument('--max_steps', type=int, help='max_steps')
    parser.add_argument('--warmup_proportion', type=float, help='warmup_proportion')
    parser.add_argument('--log_freq', type=int, help='log_freq')
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")

    args, _ = parser.parse_known_args()
    print (_)
    return args


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


if __name__ == "__main__":
    args = parse_args()
    num_nodes = args.num_nodes
    train_batch_size = args.train_batch_size
    max_seq_length = args.max_seq_length
    max_predictions_per_seq = args.max_predictions_per_seq
    max_steps = args.max_steps
    warmup_proportion = args.warmup_proportion
    log_freq = args.log_freq
    learning_rate = args.learning_rate
    seed = args.seed
    fp_16 = '--fp16' if args.fp16 else ''
    do_train = '--do_train' if args.do_train else ''
    # main_path = '/opt/ml/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT/run_pretraining.py'
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'run_pretraining.py'))
    print(main_path)

    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    # work_dir = os.environ['SM_OUTPUT_DATA_DIR']
    work_dir = '/opt/ml/code'
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    json_file = work_dir + "/dllogger.json"

    cmd = f"python -m torch.distributed.launch --nnodes={num_nodes} --node_rank={rank} --nproc_per_node={num_gpus} \
        --master_addr={hosts[0]} --master_port='12345' \
    /shared/DeepLearningExamples/PyTorch/LanguageModeling/BERT/run_pretraining.py --input_dir {data_dir} --output_dir {work_dir} --config_file /opt/ml/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT/bert_config.json --bert_model bert-large-uncased --train_batch_size {train_batch_size} --max_seq_length {max_seq_length} --max_predictions_per_seq {max_predictions_per_seq} --max_steps {max_steps} --warmup_proportion {warmup_proportion} --log_freq {log_freq} --num_steps_per_checkpoint {args.num_steps_per_checkpoint} --learning_rate {learning_rate} --seed {seed} {fp_16} {do_train} --json-summary {json_file}"

    print (cmd)
    invoke_train(cmd)
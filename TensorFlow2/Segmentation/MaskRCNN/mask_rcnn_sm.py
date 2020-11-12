# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Scripts that simplifies running training benchmark """

import argparse
import os
import shutil
import subprocess
from argparse import ArgumentError
'''
The mask_rcnn_main.py expects args with a `=` sign in between them. eg : python mask_rcnn_main.py --train_steps=100. 
This files takes parameters from SageMaker job and massages them with `=`.
'''

def csv_int(vstr, sep=','):
    values = []
    for v0 in vstr.split(sep):
        try:
            v = int(v0)
            values.append(v)
        except ValueError as err:
            raise ArgumentError('Invalid value %s, values must be a number' % v)
    return values

def main():
    # CLI flags
    parser = argparse.ArgumentParser(description="MaskRCNN train benchmark")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--eval_samples', type=int, required=True)
    parser.add_argument('--init_learning_rate', type=float, required=True)
    parser.add_argument('--learning_rate_steps', type=csv_int, required=True)
    parser.add_argument('--model_dir_ckpt', type=str, required=True)
    parser.add_argument('--num_steps_per_eval', type=int, required=True)
    parser.add_argument('--total_steps', type=int, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--eval_batch_size', type=int, required=True)
    parser.add_argument('--training_file_pattern', type=str, required=True)
    parser.add_argument('--validation_file_pattern', type=str, required=True)
    parser.add_argument('--val_json_file', type=str, required=True)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--use_batched_nms', action='store_true')
    parser.add_argument('--xla', action='store_true')
    parser.add_argument('--eval_after_training', action='store_true')
    parser.add_argument('--nouse_custom_box_proposals_op', action='store_true')
    parser.add_argument('--seed', type=int, default=987)

    flags, unknown = parser.parse_known_args()
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mask_rcnn_main.py'))

    lr_steps = ','.join([str(x) for x in flags.learning_rate_steps])
    train_files = flags.training_file_pattern + '/train*.tfrecord'
    val_files = flags.validation_file_pattern + '/val*.tfrecord'

    # build command
    cmd = (
        f'python {main_path}'
        f' --mode={flags.mode}'
        f' --checkpoint={flags.checkpoint}'
        f' --eval_samples={flags.eval_samples}'
        f' --init_learning_rate={flags.init_learning_rate}'
        f' --learning_rate_steps={lr_steps}'
        f' --model_dir={flags.model_dir_ckpt}'
        f' --num_steps_per_eval={flags.num_steps_per_eval}'
        f' --total_steps={flags.total_steps}'
        f' --train_batch_size={flags.train_batch_size}'
        f' --eval_batch_size={flags.eval_batch_size}'
        f' --training_file_pattern={train_files}'
        f' --validation_file_pattern={val_files}'
        f' --val_json_file={flags.val_json_file}'
        f' {"--amp" if flags.amp else ""}'
        f' {"--use_batched_nms" if flags.use_batched_nms else ""}'
        f' {"--xla" if flags.xla else ""}'
        f' {"--eval_after_training" if flags.eval_after_training else ""}'
        f' {"--nouse_custom_box_proposals_op" if flags.nouse_custom_box_proposals_op else ""}'
        f' --seed={flags.seed}'
    )

    # print command
    line = '-' * shutil.get_terminal_size()[0]
    print(line, cmd, line, sep='\n')

    # run model
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()

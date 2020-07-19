from evaluation.face_recognition.model import create_model

import argparse
import importlib
from evaluation.face_recognition.model import create_model

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--direction', type=str, default='AtoB', help='')
parser.add_argument('--dataroot', type=str, required=True, help='')
parser.add_argument('--max_dataset_size', type=float, default=float('inf'), help='')
parser.add_argument('--phase', type=str, default='train', help='')
parser.add_argument('--num_threads', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=4, help='')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_directory', type=str, default='evaluation/face_recognition/models')
parser.add_argument('--load_size', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=64)
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_classes', type=int, required=True)
#unused
parser.add_argument('--output_nc', type=int, default=3)

opts = parser.parse_args()

# set gpu ids
str_ids = opts.gpu_ids.split(',')
opts.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opts.gpu_ids.append(id)

create_model(opts)
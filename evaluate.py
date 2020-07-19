#!/usr/bin/env python

import argparse, importlib, torch, json, os
from evaluation.EvaluatorBase import EvaluatorBase
from collections.abc import Iterable
import numpy as np

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--epoch', type=str, default='latest', help='all for all available epochs')
parser.add_argument('--results_dir', type=str, default='./results/', help='')
parser.add_argument('--result', required=True, type=str, help='')
parser.add_argument('--metric', type=str, default='inception_score', help='')
parser.add_argument('--size', type=str, default='64x64', help='')
parser.add_argument('--num_threads', type=int, default=1, help='')
parser.add_argument('--serial_batches', type=bool, default=False, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')

parser.add_argument('--phase', type=str, default='evaluate', help='')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--model_directory', type=str, default='evaluation/face_recognition/models')
parser.add_argument('--model_name', type=str, required=True)

opts = parser.parse_args()

opts.size = tuple([int(x) for x in opts.size.split('x')])

torch.no_grad()
# set gpu ids
str_ids = opts.gpu_ids.split(',')
opts.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opts.gpu_ids.append(id)

def find_evaluator_using_name(evaluator_name):
    evaluator_filename = 'evaluation.' + evaluator_name + ".evaluator"
    evaluatorlib = importlib.import_module(evaluator_filename)
    evaluator = None
    target_evaluator_name = evaluator_name.replace('_', '')
    for name, cls in evaluatorlib.__dict__.items():
        if name.lower() == target_evaluator_name.lower() \
           and issubclass(cls, EvaluatorBase):
            evaluator = cls

    if evaluator is None:
        print("In %s.evaluator.py, there should be a subclass of EvaluatorBase with class name that matches %s in lowercase." % (evaluator_filename, target_evaluator_name))
        exit(0)

    return evaluator


evaluator = find_evaluator_using_name(opts.metric)(opts)


# fix for numpy to json: see https://interviewbubble.com/typeerror-object-of-type-float32-is-not-json-serializable/
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if opts.epoch == 'all':
    results = []
    for (epoch, _), dataset, dataloader in zip(evaluator.input, evaluator.dataset, evaluator.dataloader):
        top1, top2, top3, hits1, hits2, hits3 = evaluator.evaluate(dataset, dataloader)
        results.append((epoch, top1, top2, top3, hits1, hits2, hits3))
    with open(os.path.join(os.path.join(opts.results_dir, opts.result), 'evaluator_%s.json' % (opts.model_name)), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)

else:
    top1, top2, top3, hits1, hits2, hits3 = evaluator.evaluate(evaluator.dataset, evaluator.dataloader)
    print(top1, top2, top3, hits1, hits2, hits3)
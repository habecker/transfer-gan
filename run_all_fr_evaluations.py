#!/usr/bin/env python
import argparse, os, glob

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--results_dir', type=str, default='./results/', help='')
parser.add_argument('--model_name', type=str, required=True)

opts = parser.parse_args()

for fp in glob.glob(os.path.join(opts.results_dir, '*/')):
    results_name = os.path.basename(fp[:-1])
    if 'cityscapes' in results_name or 'old' in results_name:
        continue
    if not os.path.exists(os.path.join(fp, 'evaluator_{}.json'.format(opts.model_name))):
        print('./evaluate.py --metric face_recognition --result %s --model_name %s --epoch all' % (results_name, opts.model_name))
        os.system('./evaluate.py --metric face_recognition --result %s --model_name %s --epoch all' % (results_name, opts.model_name))
# ./evaluate.py --metric face_recognition --result celebA_edges_pretrained_sketch_feret_t4_2_pix2pix --model_name 20_lr0.002_bs8 --epoch all
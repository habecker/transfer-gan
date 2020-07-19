import glob
import re
import yaml
import os
from collections import OrderedDict

ignore_files = {'./train_classifier.sh'}
options_set = set()
options_possibilities = {}
options_for_file = OrderedDict()
experiments = []

train_regex = re.compile(r'^[ ]*python3? train\.py[ ]*(?P<options>.*)$')
test_regex = re.compile(r'^[ ]*python3? test\.py[ ]*(?P<options>.*)$')
split_regex = re.compile(r'[ ]+')
replace_regex = re.compile(r'[ ]+--')
def normalize_dataset(dataset_basename):
    if dataset_basename == 'sketch_face_64' or dataset_basename == 'sketch_face_128':
        return dataset_basename
    if dataset_basename == 'celebA_edges':
        return 'celeba_edges_64'
    elif dataset_basename == 'celebA_edges_128':
        return 'celeba_edges_128'
    elif dataset_basename == 'ImageNetAA':
        return 'imagenet_imagenet_64'
    elif dataset_basename == 'imagenet_64':
        return 'imagenet_imagenet_64'
    elif dataset_basename == 'imagenet_64_2pair':
        return 'imagenet_imagenet_64'
    elif dataset_basename == 'cityscapes_128':
        return 'labelmap_cityscapes_128'
    elif dataset_basename == 'cityscapes_64':
        return 'labelmap_cityscapes_64'
    elif dataset_basename == 'cityscapes':
        assert False
    print(dataset_basename)
    assert False
def normalize_options(options):
    new_options = OrderedDict()
    for k,v in options.items():
        if k == 'dataroot':
            new_options['dataset'] = normalize_dataset(os.path.basename(v))
            continue
        elif k == 'name':
            continue
        elif k == 'direction':
            continue
        elif k == 'input_nc':
            continue
        elif k == 'output_nc':
            continue
        elif k == 'load_size':
            continue
        elif k == 'crop_size':
            continue
        elif k == 'display_winsize':
            continue
        elif k == 'gpu_ids':
            continue
        new_options[k] = v
    return new_options

def get_file_options(f):
    options_dict = OrderedDict()
    for line in f:
        match = train_regex.search(line.rstrip())
        if match:
            options = replace_regex.sub('\t', match.group('options'))
            options = [tuple(split_regex.split(opt.replace('--',''))) for opt in options.split('\t')]
            for opt in options:
                options_set.add(opt[0])
                if len(opt) > 2:
                    exit(0)
                if len(opt) > 1:
                    if opt[0] not in options_possibilities:
                        options_possibilities[opt[0]] = []
                    options_possibilities[opt[0]] = opt[1]
                    options_dict[opt[0]] = opt[1]
                else:
                    options_dict[opt[0]] = True
            return options_dict
    return None

def get_file_test_options(f):
    options_dict = OrderedDict()
    for line in f:
        match = test_regex.search(line.rstrip())
        if match:
            options = replace_regex.sub('\t', match.group('options'))
            options = [tuple(split_regex.split(opt.replace('--',''))) for opt in options.split('\t')]
            for opt in options:
                options_set.add(opt[0])
                if len(opt) > 2:
                    print(f, opt)
                    exit(0)
                if len(opt) > 1:
                    if opt[0] not in options_possibilities:
                        options_possibilities[opt[0]] = []
                    options_possibilities[opt[0]] = opt[1]
                    options_dict[opt[0]] = opt[1]
                else:
                    options_dict[opt[0]] = True
            return options_dict
    return None

for fp in glob.glob('./bash/*.sh'):
    if fp in ignore_files:
        continue
    name = os.path.basename(fp)[:-3]
    experiments.append(name)
    with open(fp, 'r') as f:
        options_for_file[name] = normalize_options(get_file_options(f))

if __name__ == "__main__":
        
    for name, options in options_for_file.items():
        print("  - %s:" % name)
        for k,v in options.items():
            print('    %s: %s' % (k,v))
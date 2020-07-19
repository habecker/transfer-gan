#!/usr/bin/env python

import yaml, glob, os, re, sys, argparse, shutil
from collections import OrderedDict
from extract_arguments import get_file_options, get_file_test_options

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--validate', action='store_true', help='')
parser.add_argument('--phase', type=str, help='train/test')
parser.add_argument('--results', type=str, default='./results/', help='')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='')
parser.add_argument('--name', type=str, required='--phase' in sys.argv, help='')
parser.add_argument('--gpu', type=int, required='--phase' in sys.argv, help='')
parser.add_argument('--cover', action='store_true', help='')
parser.add_argument('--steps', type=int, default=5, help='')
parser.add_argument('--headless', action='store_true', help='')
parser.add_argument('--test_phase', type=str, default='test', help='')


parser.add_argument('--num_test', type=int, default=None, help='how many test images to run')

parser.add_argument('--continue_gan', type=int, default=-1, help='continue gan training at epoch [0-9]+')

transfer_name_regex = re.compile('(?P<name>t[0-9]+)_(?P<variant>[0-9]+)')

with open('./experiments/experiments.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    datasets = data['datasets']
    possible_options = data['possible_options']
    experiments = data['experiments']
    transfer_experiments = data['transfer_experiments']

def make_name(name, experiment):
    dataset = datasets[experiment['dataset']]
    prefix = experiment['prefix'] if 'prefix' in experiment else dataset['prefix']
    return "{}_{}_pix2pix".format(prefix, name)

def make_options(name, experiment, gpu=-1):
    options = OrderedDict()
    for k,v in experiment.items():
        if k == 'dataset':
            dataset = datasets[v]
            options['dataroot'] = dataset['dataroot']
            options['input_nc'] = dataset['input_nc']
            options['output_nc'] = dataset['output_nc']
            options['direction'] = dataset['direction']
            options['load_size'] = dataset['image_size']
            options['crop_size'] = dataset['image_size']
            options['display_winsize'] = dataset['image_size']
            continue
        assert k not in options
        options[k] = v
    options['display_env'] = name
    options['gpu_ids'] = gpu
    options['name'] = make_name(name, experiment)
    return options

def make_transfer_options(name, gpu=-1):
    experiment = transfer_experiments[name]
    experiments = {}
    for i, variant in enumerate(experiment['variants'][1:]):
        options = {}
        for k,v in experiment.items():
            if k in {'transfer', 'transfer_with', 'variants', 'prefix'}:
                continue
            if k == 'dataset':
                dataset = datasets[v]
                options['dataroot'] = dataset['dataroot']
                options['input_nc'] = dataset['input_nc']
                options['output_nc'] = dataset['output_nc']
                options['direction'] = dataset['direction']
                options['load_size'] = dataset['image_size']
                options['crop_size'] = dataset['image_size']
                options['display_winsize'] = dataset['image_size']
                continue
            assert k not in options
            options[k] = v
        if experiment['transfer_with'] == 'gd':
            options['continue_train'] = True
        elif experiment['transfer_with'] == 'd':
            options['continue_discriminator'] = True
        for j,v in enumerate(variant):
            key = experiment['variants'][0][j]
            options[key] = v
        options['display_env'] = name + '_%d' % (i+1)
        options['gpu_ids'] = gpu
        options['name'] = make_name(name + '_%d' % (i+1), experiment)
        experiments[name + '_%d' % (i+1)] = options
    return experiments

def make_test_options(name, experiment, results_name, gpu=-1):
    options = OrderedDict()
    for k,v in experiment.items():
        if k == 'dataset':
            dataset = datasets[v]
            options['dataroot'] = dataset['dataroot']
            options['input_nc'] = dataset['input_nc']
            options['output_nc'] = dataset['output_nc']
            options['direction'] = dataset['direction']
            options['load_size'] = dataset['image_size']
            options['crop_size'] = dataset['image_size']
            options['display_winsize'] = dataset['image_size']
            continue
        if k in {'gan_mode', 'batch_size', 'display_id', 'lambda_L1', 'continue_train', 'continue_discriminator', 'niter', 'niter_decay', 'prefix', 'lr', 'display_env'}:
            continue
        assert k not in options
        options[k] = v
    options['gpu_ids'] = gpu
    options['name'] = name
    return options

def make_train_command(name, gpu=-1):
    options = make_options(name, experiments[name], gpu)
    return make_train_command_by_options(name, options, gpu)

def make_train_command_by_options(name, options, gpu):
    arguments = ''
    for k,v in options.items():
        if type(v) == bool:
            arguments += ' --{}'.format(k)
            continue
        arguments += ' --{} {}'.format(k,v)
    return 'python3 train.py' + arguments

def make_test_command(experiment_name, name, gpu=-1, is_transfer=False, transfer_options=None):
    if is_transfer:
        experiment = transfer_options
    else:
        experiment = experiments[experiment_name]
    arguments = ''
    for k,v in make_test_options(name, experiment, gpu).items():
        if k in {'load_iter'}:
            continue
        if type(v) == bool:
            arguments += ' --{}'.format(k)
            continue
        arguments += ' --{} {}'.format(k,v)
    return 'python3 test.py' + arguments

def validate():    
    for fp in glob.glob('./bash/*.sh'):
        name = os.path.basename(fp)[:-3]
        with open(fp, 'r') as f:
            target_options = get_file_options(f)
            f.seek(0)
            test_options = get_file_test_options(f)
        created_options = make_options(name, experiments[name])
        for k,v in created_options.items():
            if k in {'prefix', 'gpu_ids', 'display_env'}:
                continue
            if k not in target_options:
                print("%s: %s not in target options" % (fp, k))
        for k,v in target_options.items():
            if k in {'gpu_ids', 'display_env'}:
                continue
            if k not in created_options:
                print("%s: %s not in options" % (fp, k))
                continue
            if type(v) != type(created_options[k]):
                continue
            if v != created_options[k]:
                print("Value for key %s in created options is wrong: %s != %s" % (k, v, created_options[k]))
                continue
        wrong_test_args = set(make_test_options(name, experiments[name]).keys()) - set(test_options.keys()) - {'display_env', 'gpu_ids'}
        if len(wrong_test_args) > 0:
            print(wrong_test_args)
        print(make_train_command(name))
        print(make_test_command(name))
        #print(make_train_command(name))
    print("Validation completed")

def prepare_transfer_experiment(name, target_name, load_iter, copy_generator=False, copy_discriminator=False, checkpoints='./checkpoints'):
    experiment = transfer_experiments[name]
    pre_experiment = experiments[experiment['transfer']]
    source_name = make_name(experiment['transfer'], pre_experiment)
    source_directory = os.path.join(checkpoints, source_name)
    target_directory = os.path.join(checkpoints, target_name)
    if os.path.exists(target_directory):
        print("Error: %s already exists" % target_directory)
        exit(0)
    if not os.path.exists(source_directory):
        print("Error: %s does not exists" % source_directory)
        exit(0)
    
    os.mkdir(target_directory)
    if copy_generator:
        print("Copying Generator")
        source_pth = os.path.join(source_directory, '%s_net_G.pth' % load_iter)
        target_pth = os.path.join(target_directory, 'iter_%s_net_G.pth' % load_iter)
        if not os.path.exists(source_pth):
            print("Error: %s does not exist" % source_pth)
            exit(0)
        shutil.copyfile(source_pth, target_pth)
    if copy_discriminator:
        print("Copying Discriminator")
        source_pth = os.path.join(source_directory, '%s_net_D.pth' % load_iter)
        target_pth = os.path.join(target_directory, 'iter_%s_net_D.pth' % load_iter)
        if not os.path.exists(source_pth):
            print("Error: %s does not exist" % source_pth)
            exit(0)
        shutil.copyfile(source_pth, target_pth)
    print(make_name(experiment['transfer'], pre_experiment))
    print(target_name)

if __name__ == "__main__":
    opts = parser.parse_args()

    if opts.validate:
        validate()
        exit(0)
    elif opts.phase == 'train':
        if opts.name not in experiments:
            print("ERROR: Experiment %s does not exist" % opts.name)
            exit(1)
    
    if opts.phase == 'train':
        out_dir = os.path.join(opts.checkpoints, make_name(opts.name, experiments[opts.name]), 'train_opt.txt')
        if opts.continue_gan < 0 and os.path.exists(out_dir):
            print("ERROR: %s already exists" % out_dir)
            exit(1)
        additional_options = ''
        if opts.continue_gan > 0:
            additional_options += ' --epoch_count %d --continue_train' % opts.continue_gan
        if opts.headless:
            additional_options += ' --display_id 0'
        os.system(make_train_command(opts.name, opts.gpu) + additional_options)
    elif opts.phase == 'transfer':
        match = transfer_name_regex.match(opts.name)
        if not match:
            print("Falsely formatted transfer experiment name")
            exit(0)
        name = match.group('name')
        #out_dir = os.path.join(opts.checkpoints, make_name(opts.name, experiments[opts.name]), 'train_opt.txt')
        #if opts.continue_gan < 0 and os.path.exists(out_dir):
        #    print("ERROR: %s already exists" % out_dir)
        #    exit(1)
        # additional_options = ''
        # if opts.continue_gan:
        #     additional_options += ' --epoch_count %d --continue_train' % opts.continue_gan
        additional_options = ''
        if opts.headless:
            additional_options += ' --display_id 0'
        options = make_transfer_options(name, gpu=opts.gpu)[opts.name]
        prepare_transfer_experiment(name, options['name'], load_iter=options['load_iter'], copy_generator='continue_train' in options, copy_discriminator='continue_train' in options or 'continue_discriminator' in options, checkpoints=opts.checkpoints)
        os.system(make_train_command_by_options(opts.name, options, opts.gpu) + additional_options)
    elif opts.phase == 'test':
        match = transfer_name_regex.match(opts.name)
        is_transfer = False
        transfer_options = None
        if match:
            transfer_options = make_transfer_options(match.group('name'), gpu=opts.gpu)[opts.name]
            name = transfer_options['name']
            is_transfer = True
        else:
            name = make_name(opts.name, experiments[opts.name])
        out_dir = os.path.join(opts.results, name)
        #if os.path.exists(out_dir):
        #    print("ERROR: %s already exists" % out_dir)
        #    exit(1)
        if opts.cover:
            checkpoints_dir = os.path.join(opts.checkpoints, name)
            epochs = []
            for fp in glob.glob(os.path.join(checkpoints_dir, '*_net_G.pth')):
                if 'iter' in fp:
                    continue
                epoch = os.path.basename(fp).split('_')[0]
                if epoch != 'latest' and int(epoch) % opts.steps == 0:
                    epochs.append(int(epoch))
            epochs = sorted(epochs) + ['latest']
            for epoch in epochs:
                ret = os.system(make_test_command(opts.name, name, opts.gpu, is_transfer=is_transfer, transfer_options=transfer_options) + ' --epoch {}'.format(epoch) + (' --phase train' if opts.test_phase == 'train' else '') + (' --num_test {}'.format(opts.num_test) if opts.num_test is not None else ''))
                if ret != 0:
                    exit(1)
                    break
        else:
            os.system(make_test_command(opts.name, name, opts.gpu) + (' --phase train' if opts.test_phase == 'train' else '') + (' --num_test {}'.format(opts.num_test) if opts.num_test is not None else ''))

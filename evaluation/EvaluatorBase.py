import os
import glob
from data.image_folder import ImageFolder, IMG_EXTENSIONS, default_loader, is_image_file
from PIL import Image
import torchvision.transforms as transforms
import torch

def get_transform(resize=False, padding=None, osize=(299, 299), method=Image.BICUBIC, convert=True):
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(osize, method))
    
    if padding:
        transform_list.append(transforms.Pad(padding, fill=(0,0,0)))

    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def make_results_dataset(dir, max_dataset_size=float("inf"), filter='*fake*'):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for fp in glob.glob(os.path.join(dir, filter)):
        if is_image_file(os.path.basename(fp)):
            images.append(fp)
    return images[:min(max_dataset_size, len(images))]

class ResultsImageFolder(ImageFolder):
    def __init__(self, root, transform=None, return_paths=False):
        imgs = make_results_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = default_loader

class EvaluatorBase(object):
    def __init__(self, opts, transform=None):
        if opts.epoch != 'all':
            self.input = os.path.join(opts.results_dir, opts.result)
            if not os.path.exists(self.input):
                print("%s does not exist" % self.input)
                exit(0)
        else:
            self.input = [(int(os.path.basename(p[:-1])[5:]), p) for p in glob.glob(os.path.join(opts.results_dir, opts.result, 'test_*/')) if 'latest' not in p] # tuple of epoch number and path
        
        self.transform = transform
        self.load_images(opts)

    def load_images(self, opts):
        if type(self.input) is list:
            self.dataset = []
            self.dataloader = []
            for inp in self.input:
                dataset = ResultsImageFolder(self.input, transform=self.transform)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opts.batch_size,
                    shuffle=not opts.serial_batches,
                    num_workers=int(opts.num_threads))
                self.dataset += [dataset]
                self.dataloader += [dataloader]
        else:
            self.dataset = ResultsImageFolder(self.input, transform=self.transform)

            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opts.batch_size,
                shuffle=not opts.serial_batches,
                num_workers=int(opts.num_threads))

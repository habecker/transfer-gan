from evaluation.EvaluatorBase import EvaluatorBase, get_transform
from torch.autograd import Variable
import numpy as np
import torch
from torch.nn import functional as F
from evaluation.face_recognition.model import create_model
import os
import glob
from PIL import Image

class ResultsLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, results_dir, opt, dataset_dir, contain_persons, transform=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_meta(os.path.join('./datasets', dataset_dir, 'meta_test.txt'))
        self.images_dir = os.path.join(results_dir, 'images')  # get the image directory
        image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*fake*.png')))  # get image paths
        self.contain_persons = sorted(contain_persons)
        #individuals = sorted(set([m[0] for m in self.meta]))
        print(results_dir,len(self.meta),len(image_paths))
        assert(len(self.meta) == len(image_paths))
        labels = [None]*len(image_paths)
        self.transform = transform
        self.labels = []
        self.image_paths = []
        for index, _ in enumerate(image_paths):
            if self.meta[index][0] in self.contain_persons:
                self.image_paths.append(image_paths[index])
                self.labels.append(self.contain_persons.index(self.meta[index][0]))
                # print(image_paths[index], self.meta[index][0])

    def load_meta(self, path):
        meta = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = line.split(',')
                data = [line[1]] + [float(s) for s in line[2:]]
                meta[int(line[0])] = data
        self.meta = meta

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a tuple that contains label, image
        """
        # read a image given a random integer index
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, torch.tensor(self.labels[index]))#{'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'meta': self.meta[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

import yaml


def deactivate_track_running_stats(model):
    for child in model.modules():
        if type(child) == torch.nn.BatchNorm2d:
            child.track_running_stats = False
    model.eval()

class FaceRecognition(EvaluatorBase):
    def __init__(self, opts):
        parameters = {}
        with open(os.path.join(opts.model_directory, opts.model_name[:2].replace('_','') + '.yaml'), 'r') as f:
            parameters = yaml.safe_load(f)
        self.model_dataset = parameters['dataset']
        opts.num_classes = len(parameters['classes'])
        self.classes = parameters['classes']

        super().__init__(opts, get_transform())

        self.recognition_net = create_model(opts)
        # deactivate_track_running_stats(self.recognition_net)
        self.batch_size = opts.batch_size
        self.num_classes = opts.num_classes

    def get_pred(self, x):
        x = self.recognition_net(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # def get_target_tensor(self, targets):
    #     tensor = np.zeros(())

    def evaluate(self, dataset, dataloader, cuda=False):
        # Set up dtype
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor
        score = 0.0
        score_top2 = 0.0
        score_top3 = 0.0

        hits = []
        hits_top2 = []
        hits_top3 = []

        for i, batch in enumerate(dataloader, 0):
            target = batch[1].type(dtype)
            batch = batch[0].type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            predictions = self.get_pred(batchv)
            # predictions = np.argmax(predictions, 1)
            for a,b in zip(target, predictions):
                a = int(a)
                p = np.argmax(b)
                p2s = b.argsort()[-2:][::-1]
                score_top2 += 1.0 if a in p2s else 0.0
                p3s = b.argsort()[-3:][::-1]
                score_top3 += 1.0 if a in p3s else 0.0
                score += 1.0 if a == p else 0.0

                if a == p:
                    hits += [(a, b[p])]
                if a in p2s:
                    hits_top2 += [(a, b[p])]
                if a in p3s:
                    hits_top3 += [(a, b[p])]

        return (score/np.float(len(dataset)), score_top2/np.float(len(dataset)), score_top3/np.float(len(dataset)), hits, hits_top2, hits_top3)

    def load_images(self, opts):
        if type(self.input) is list:
            self.dataset = []
            self.dataloader = []
            for epoch, inp in self.input:
                dataset = ResultsLabeledDataset(inp, opts, self.model_dataset, transform=self.transform, contain_persons=self.classes)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opts.batch_size,
                    shuffle=not opts.serial_batches,
                    num_workers=int(opts.num_threads))
                self.dataset += [dataset]
                self.dataloader += [dataloader]
        else:
            self.dataset = ResultsLabeledDataset(os.path.join(self.input, 'test_%s' % opts.epoch), opts, self.model_dataset, transform=self.transform, contain_persons=self.classes)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opts.batch_size,
                shuffle=not opts.serial_batches,
                num_workers=int(opts.num_threads))


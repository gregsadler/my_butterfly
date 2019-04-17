import torch
import torch.utils.data
import torchvision
from pathlib import Path
from collections import defaultdict
import random
from itertools import groupby, chain
from PIL import Image
import numpy as np
import math
import random
import utils.data_prep as DP


class RandomRotation(object):
    '''Random rotation of the image. The image is first padded using reflect
    mode, then rotated, then center cropped to the original size. This
    prevents the output from having black bands in the corners.
    '''
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, image):
        degree = random.random() * 2 * self.theta - self.theta
        w, h = image.size
        pad = int((math.sqrt(2) - 1) * (w // 2))
        image = torchvision.transforms.functional.pad(
                image, (pad, pad), padding_mode='reflect')
        image = image.rotate(degree, Image.BILINEAR)
        image = torchvision.transforms.functional.center_crop(image, (w, h))

        return image


def _normalized_tensor_transform():
    # Imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    T = torchvision.transforms
    tensor_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std) 
    ])
    
    return tensor_normalize


class ButterflyDataset(torch.utils.data.Dataset):

    def __init__(self, root, train=True, size=224, augment=True):

        self.root = Path(root) / ('train' if train else 'test')
        self.size = size
        self.augment = augment

        self.imagepaths = []
        self.classes = []
        for class_ in self.root.iterdir():
            if class_.is_dir():
                self.classes.append(class_.name)
                for img in class_.iterdir():
                    self.imagepaths.append(img)

        self.classes.sort()
        self.class_to_idx = {c: i for i,c in enumerate(self.classes)}

        if train and augment:
            self.augment = self._augmentation(size)
        else:
            self.augment = torchvision.transforms.Resize((size, size))
        self.tensor_normalize = _normalized_tensor_transform()

    def _augmentation(self, size):
        T = torchvision.transforms
        a = T.Compose([
            T.RandomResizedCrop(size, (0.8, 1.0), (4/5, 5/4)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])
        return a

    def __getitem__(self, index):
        path = self.imagepaths[index]
        target = self.class_to_idx[path.parts[-2]]

        img = Image.open(path).convert('RGB')

        img = self.augment(img)
        img = self.tensor_normalize(img)

        return img, target

    def __len__(self):
        return len(self.imagepaths)


def _wing_data(root):
    classes = []
    data = []
    class_part = {}
    i = 0
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        spec = class_dir.name
        classes.append(spec)
        class_part[spec] = defaultdict(list)

        for part_img in class_dir.iterdir():
            img, part = part_img.name.split('-')
            part = part.split('.')[0]
            data.append((spec, img, part))
            class_part[spec][part].append(i)
            i += 1

    classes = sorted(classes)
    class_num = {classes[i]: i for i in range(len(classes))}

    return classes, class_num, data, class_part


class WingDataset(torch.utils.data.Dataset):
          
    def __init__(self, root, train=True, parts= False):
        super(WingDataset, self).__init__()
        self.root = Path(root).joinpath('train' if train else 'test')
        self.train = train
        self.part_labels = parts

        classes, class_num, data, class_part = _wing_data(self.root)
        self.classes = classes
        self.class_num = class_num
        self.data = data
        self.class_part = class_part


        self.part_id = {
            'rvh': 0, 'lvh': 0, 'rvf': 1, 'lvf': 1,
            'rdh':2, 'ldh': 2, 'rdf': 3, 'ldf': 3,
        }

        if train:
            self.augment = self._augmentation()
        else:
            self.augment = torchvision.transforms.Resize((224, 224))
        self.tensor_normalize = _normalized_tensor_transform()

    def __len__(self):
        return len(self.data)

    def _augmentation(self):
        T = torchvision.transforms
        a = T.Compose([
            T.RandomResizedCrop(224, (0.8, 1.0), (4/5, 5/4)),
            T.RandomHorizontalFlip(),
            RandomRotation(180),
        ])
        return a
    
    def __getitem__(self, index):
        spec, img_id, part = self.data[index]
        img_file = self.root / spec / (img_id + '-' + part + '.png')
        img = Image.open(img_file).convert('RGB')
        img = self.augment(img)
        img = self.tensor_normalize(img)
        label = self.class_num[spec]
        if self.part_labels:
            return img, label, self.part_id[part] 
        else:
            return img, self.part_id[part]


def _wing_seg_data(root, split='train'):
    
    mask_loc = Path('/multiview/datasets/papillon/full_crops/masks_generated')
    pose_loc = Path('/multiview/datasets/papillon/images_by_pose')

    classes = []
    data = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        spec = class_dir.name
        classes.append(spec)
        imgdata_for_class = {}
        for part_img in class_dir.iterdir(): #this is where I can filter bad images
            img, part = part_img.name.split('-')
            part = part.split('.')[0]
            if img not in imgdata_for_class:
                imgdata_for_class[img] = []
            imgdata_for_class[img].append((spec, img, part))
        # Add in all of the images
        for imgid in imgdata_for_class:
            part_list = imgdata_for_class[imgid]
            spec,img,part = part_list[0]
            
            pose_dir = pose_loc / spec
            pose = "none"
            for pose_ in pose_dir.iterdir():
                if (pose_dir/pose_/(imgid+'.jpg')).exists():
                    pose = str(pose_.parts[-1])
                    pose.replace('\n', '')
            seg_file = mask_loc / split / (imgid + '.npz')
            data.append((pose, seg_file, imgdata_for_class[imgid]))

    classes = sorted(classes)
    class_num = {classes[i]: i for i in range(len(classes))}

    return classes, class_num, data


def _build_part_index(data, class_id, part_id):
    part_index = {}
    ii = 0
    all_data = []
    for i, entry in enumerate(data):
        _,_, partlist = entry
        for spec, img_id, part_str in partlist:
            fname = '{}/{}-{}.png'.format(spec, img_id, part_str)
            cid, pid = class_id[spec], part_id[part_str]
            all_data.append((cid, pid, fname))

            if cid not in part_index:
                part_index[cid] = {}
            if pid not in part_index[cid]:
                part_index[cid][pid] = []
            part_index[cid][pid].append(ii)
            ii += 1

    return all_data, part_index


class InvariantWingDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, num_trans=9, n_pos=3, n_neg=3):
        super(InvariantWingDataset, self).__init__()

        self.root = Path(root).joinpath('train' if train else 'test')
        self.train = train

        self.part_id = {
            'rvh': 0, 'lvh': 0, 'rvf': 1, 'lvf': 1,
            'rdh':2, 'ldh': 2, 'rdf': 3, 'ldf': 3,
        }

        self.num_trans = num_trans
        self.n_pos = n_pos
        self.n_neg = n_neg

        classes, class_num, data = _wing_seg_data(self.root)
        self.part_data, self.part_index = (
            _build_part_index(data, class_num, self.part_id))
        self.classes = classes
        self.class_num = class_num

        if train:
            self.augment = self._augmentation()
        else:
            self.augment = torchvision.transforms.Resize((224, 224))
        self.tensor_normalize = _normalized_tensor_transform()

    def __len__(self):
        return len(self.part_data)

    def _augmentation(self):
        T = torchvision.transforms
        trans = T.Compose([
            T.RandomResizedCrop(224, (0.8, 1.1), (4/5, 5/4)),
            T.RandomHorizontalFlip(),
            RandomRotation(180),
            #T.ColorJitter(0.1, 0.1, 0.1), # TODO
        ])
        return trans

    def __getitem__(self, index):
        pos_cid, pid, fname = self.part_data[index]
        nclasses = len(self.classes)

        # sample images from the positive set
        cand_pos_images = set(np.int16(self.part_index[pos_cid][pid]))
        cand_pos_images.remove(index)

        n_other = self.n_pos - 1
        if len(cand_pos_images) == 0:
            # no other images available - replicate this one
            other_pos = [index] * n_other
        else:
            # randomly sample from available images - with replacement if
            # there aren't enough
            too_few = len(cand_pos_images) < n_other
            other_inds = np.random.choice(
                len(cand_pos_images), n_other, replace=too_few)
            other_pos = np.asarray(
                sorted(list(cand_pos_images)))[other_inds]

        # sample images from the negative set
        cand_neg_cats = set([
            c for c in range(nclasses)
            if pid in self.part_index[c]
        ])
        cand_neg_cats.remove(pos_cid)
        neg_cat_inds = np.random.choice(
            len(cand_neg_cats), self.n_neg, replace=False)
        neg_cats = np.asarray(
            sorted(list(cand_neg_cats)))[neg_cat_inds]
        negs = [
            self.part_index[nc][pid][np.random.choice(
                len(self.part_index[nc][pid]), 1, replace=False)[0]]
            for nc in neg_cats
        ]

        # load and process the images
        all_inds = [index] + [p for p in other_pos] + negs
        img_list = []
        class_list = []
        for _cid, _pid, fname in [self.part_data[ii] for ii in all_inds]:
            img = Image.open(self.root / fname).convert('RGB')
            imgs = [self.augment(img) for _ in range(self.num_trans)]
            img = torchvision.transforms.functional.resize(img, (224, 224))
            imgs = [img] + imgs
            imgs = [self.tensor_normalize(img) for img in imgs]
            img_list.extend(imgs)
            class_list.extend([_cid for _ in range(len(imgs))]) 

        # put the data into tensors
        img_tensor = torch.stack(img_list,  0)
        class_list = torch.from_numpy(np.asarray(class_list))
        parts = torch.LongTensor([pid] * class_list.shape[0])
        
        return img_tensor, class_list, parts


class SegStatsAndWingsDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train=True):
        super(SegStatsAndWingsDataset, self).__init__()
        split = 'train' if train else 'test'
        self.root = Path(root) / split
        self.train = train

        classes, class_num, data = _wing_seg_data(self.root, split)
        self.classes = classes
        self.class_num = class_num
        self.data = data

        self.seg_stats = {}

        if train:
            self.augment = self._augmentation()
        else:
            self.augment = torchvision.transforms.Resize((224, 224))
        self.tensor_normalize = _normalized_tensor_transform()

        self.part_index = {
            'rvh': 0, 'lvh':4, 'rvf': 1, 'lvf':5,
            'rdh':2, 'ldh': 6, 'rdf': 3, 'ldf': 7,
        }
        
        self.poses = {
                'dorsal_full': 0, 'dorsal_left': 1,  'dorsal_left_ventral_right':2 , 'dorsal_right':3,  
                'dorsal_right_ventral_left':4, 'ventral_full':5, 'ventral_left':6, 'ventral_right':7, 'none':8 }


    def __len__(self):
        return len(self.data)

    def _augmentation(self):
        T = torchvision.transforms
        trans = T.Compose([
            T.RandomResizedCrop(224, (0.8, 1.1), (4/5, 5/4)),
            T.RandomHorizontalFlip(),
            RandomRotation(180),
        ])
        return trans

    def __getitem__(self, index):
        pose, segfile, partlist = self.data[index]
        if segfile.name not in self.seg_stats:
            np_seg_stats, np_parts_present =DP.seg_stats_from_npz(segfile) 
            seg_stats = torch.from_numpy(np.float32(np_seg_stats).flatten())
            valid_part_mask = torch.from_numpy(np.float32(np_parts_present).flatten())[:-1] #[1:]
            self.seg_stats[segfile.name] = [seg_stats, valid_part_mask]
        else:
            seg_stats, valid_part_mask = self.seg_stats[segfile.name]


        part_imgs = []
        part_ids = []
        
        for spec, img_id, part_id in partlist:
            if valid_part_mask[self.part_index[part_id]] == 0:
                continue
       
            img_file = self.root / spec / (img_id + '-' + part_id + '.png')
            img = Image.open(img_file).convert('RGB')
            img = self.tensor_normalize(self.augment(img))
            class_label = self.class_num[spec]

            part_imgs.append(img)
            part_ids.append(part_id)
        if len(part_ids) == 0:
            print(partlist)
        part_imgs = torch.stack(part_imgs, 0)
        pose_r = self.poses[pose] 
        return seg_stats, part_imgs, class_label, part_ids, valid_part_mask, pose_r


# ******************************************************** #
# **************** COLLATE FUNCTIONS ********************* #
# ******************************************************** #

def multi_collate(samples):
    c = []
    for i in range(len(samples[0])):
        c.append(torch.cat([s[i] for s in samples], 0))
    return c


def wing_collate(samples):
    return [
        torch.cat([s[0] for s in samples], 0),
        torch.LongTensor([s[1] for s in samples]),
        torch.LongTensor([s[2] for s in samples])
    ]


def seg_multiwing_collate(samples):
    #PART_INDEX_0 = {'b':0,'ldf':1, 'ldh':2, 'lvf':3, 'lvh':4,
    #                'rdf':5, 'rdh':6, 'rvf':7, 'rvh':8}
    #PART_INDEX_OF = {
    #        'rdf': 0, 'ldf': 0, 'rdh': 1, 'ldh': 1,
    #        'rvf': 2, 'lvf': 2, 'rvh': 3, 'lvh': 3,
    #    }


    PART_INDEX_OF_ = {
            'rvh': 0, 'lvh': 0, 'rvf': 1, 'lvf': 1,
            'rdh':2, 'ldh': 2, 'rdf': 3, 'ldf': 3,
        }


    PART_INDEX = {
            'rvh': 0, 'lvh':4, 'rvf': 1, 'lvf':5,
            'rdh':2, 'ldh': 6, 'rdf': 3, 'ldf': 7,
        }


    stats     = [ x[0] for x in samples ]
    part_sets = [ x[1] for x in samples ]
    labels    = [ x[2] for x in samples ]
    labels_t  = [ torch.LongTensor([x]) for x in labels ]
    part_ids  = [ x[3] for x in samples ]
    valpmasks  = [ x[4] for x in samples ]
    poses = [x[5]for x in samples]
    poses_t = [ torch.LongTensor([x]) for x in poses]


    part_ids_f= list(chain.from_iterable(part_ids))
    part_ids_t= [ torch.Tensor([PART_INDEX[x]]) for x in part_ids_f ]
    part_cnts = [ torch.Tensor([len(x)]) for x in part_ids ]
    stats_c   = torch.stack(stats,0)
    parts_c   = torch.cat(part_sets,0)

    poses_c = torch.stack(poses_t,0)
    labels_c  = torch.stack(labels_t,0)
    partcnts_c= torch.stack(part_cnts,0)
    partids_c = torch.stack(part_ids_t,0).long()
    validpartmasks_c = torch.stack(valpmasks,0)
    return stats_c, parts_c, labels_c, partids_c, partcnts_c, validpartmasks_c, poses_c


class RandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, size, sub_size):
        self.size = size
        self.sub_size = sub_size

    def __iter__(self):
        return (i for i in random.sample(range(self.size), self.sub_size))

    def __len__(self):
        return self.sub_size


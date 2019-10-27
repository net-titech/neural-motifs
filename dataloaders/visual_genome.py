"""
File that involves dataloaders for the Visual Genome dataset.
"""
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from config import stanford_path
from pathlib import Path
from ast import literal_eval
import json

LOAD_IMAGE = True
ZSL_SPLIT_FN = stanford_path('zsl_split.dict')

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

from collections import defaultdict
from pycocotools.coco import COCO

if LOAD_IMAGE:
    from PIL import Image
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    from dataloaders.blob import Blob
    from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
        RandomOrder, Hue, random_crop

class VG(Dataset):

    splitter = None

    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 use_proposals=False, split_mask=None):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
            split_mask=split_mask
        )

        self.filenames = load_image_filenames(image_file)
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

        if use_proposals:
            print("Loading proposals", flush=True)
            p_h5 = h5py.File(PROPOSAL_FN, 'r')
            rpn_rois = p_h5['rpn_rois']
            rpn_scores = p_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(p_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np.column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        if LOAD_IMAGE:
            tform = [
                SquarePad(),
                Resize(IM_SCALE),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    @classmethod
    def splits_zsl_detector(cls, set_num, n_test=10, n_fold=5, n_val=10, *args, **kwargs):
        if cls.splitter is None:
            cls.splitter = SplitZSL(n_test=n_test, n_fold=n_fold, n_val=n_val)
        train_mask = cls.splitter.get_train_mask(set_num)
        train = cls(mode='train', split_mask=train_mask, *args, **kwargs)
        val = cls(mode='val', split_mask=train_mask, *args, **kwargs)
        test = cls(mode='test', split_mask=train_mask, *args, **kwargs)
        return train, val, test

    @classmethod
    def splits_zsl_learner(cls, set_num, n_test=10, n_fold=5, n_val=10, *args, **kwargs):
        if cls.splitter is None:
            cls.splitter = SplitZSL(n_test=n_test, n_fold=n_fold, n_val=n_val)
        train_mask = cls.splitter.get_train_mask(set_num)
        train = cls(mode='test', split_mask=train_mask, num_val_im=0, *args, **kwargs)
        val_mask = cls.splitter.get_val_mask(set_num)
        val = cls(mode='test', split_mask=val_mask, num_val_im=0, *args, **kwargs)
        test_mask = cls.splitter.get_test_mask(set_num)
        test = cls(mode='test', split_mask=test_mask, num_val_im=0, *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):

        if LOAD_IMAGE:
            image_unpadded = Image.open(self.filenames[index]).convert('RGB')
            
            # Optionally flip the image if we're doing training
            gt_boxes = self.gt_boxes[index].copy()
            flipped = self.is_train and np.random.random() > 0.5
            # Boxes are already at BOX_SCALE
            if self.is_train:
                # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                    None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                    None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

                # # crop the image for data augmentation
                # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

            w, h = image_unpadded.size
            box_scale_factor = BOX_SCALE / max(w, h)

            if flipped:
                scaled_w = int(box_scale_factor * float(w))
                # print("Scaled w is {}".format(scaled_w))
                image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
                gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

            img_scale_factor = IM_SCALE / max(w, h)
            if h > w:
                im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
            elif h < w:
                im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
            else:
                im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)

        if LOAD_IMAGE:
            entry = {
                'img': self.transform_pipeline(image_unpadded),
                'img_size': im_size,
                'gt_boxes': gt_boxes,
                'gt_classes': self.gt_classes[index].copy(),
                'gt_relations': gt_rels,
                'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
                'index': index,
                'flipped': flipped,
                'fn': self.filenames[index],
            }
        else:
            entry = {
                'img': None,
                'img_size': None,
                'gt_boxes': self.gt_boxes[index].copy(),
                'gt_classes': self.gt_classes[index].copy(),
                'gt_relations': gt_rels,
                'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
                'index': index,
                'flipped': None,
                'fn': self.filenames[index],
            }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        if LOAD_IMAGE:
            assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if not LOAD_IMAGE or os.path.exists(filename):
            fns.append(filename)
    assert len(fns) == 108073
    return fns


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False, split_mask=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0

    if split_mask is None:
        split_mask = data_split == split

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load


def get_max(array, k=10):
    indexes = np.where(array > float('-inf'))
    values = array[indexes]
    ind = np.argpartition(values, -k)[-k:]
    ind = ind[np.argsort(values[ind])][::-1]
    return [index[ind] for index in indexes], values[ind]


class SplitZSL:

    def __init__(self,  load=True, *args, **kwargs):
        global LOAD_IMAGE
        _old_setting = LOAD_IMAGE
        LOAD_IMAGE = False
        split_mask_1, gt_boxes_1, gt_classes_1, relationships_1 = load_graphs(VG_SGG_FN, mode='train', filter_empty_rels=False)
        split_mask_2, gt_boxes_2, gt_classes_2, relationships_2 = load_graphs(VG_SGG_FN, mode='test', filter_empty_rels=False)
        # split_mask_3, gt_boxes_3, gt_classes_3, relationships_3 = load_graphs(VG_SGG_FN, mode='test')

        self.split_mask = split_mask_1 | split_mask_2
        # gt_boxes = gt_boxes_1 + gt_boxes_2 + gt_boxes_3
        self.gt_classes = gt_classes_1 + gt_classes_2
        self.relationships = relationships_1 + relationships_2

        self.i2c, self.i2p = load_info(VG_SGG_DICT_FN)
        filenames = load_image_filenames(IM_DATA_FN)
        self.filenames = [filenames[i] for i in np.where(self.split_mask)[0]]

        self.obj_count = np.zeros(len(self.i2c))
        for classes in self.gt_classes:
            self.obj_count[np.unique(classes)] += 1
        if load and os.path.exists(ZSL_SPLIT_FN):
            with open(ZSL_SPLIT_FN) as f:
                self.load_dict = literal_eval(f.read())
                for key, value in self.load_dict.items():
                    setattr(self, key, value)
        else:
            self.hold_test_classes(*args, **kwargs)
        LOAD_IMAGE = _old_setting

    def hold_test_classes(self, n_test=10, n_fold=5, n_val=10):
        test_classes_pool = range(len(self.i2c))
        test_classes_pool = get_max(-self.obj_count, k=n_test*n_fold)[0][0]
        # test_classes = np.random.choice(test_classes_pool, size=15, replace=False)
        np.random.shuffle(test_classes_pool)
        self.test_classes_list = [list(test_classes_pool[i*n_test: (i+1)*n_test]) for i in range(n_fold)]
        self.val_classes_list = []
        for test_classes in self.test_classes_list:
            rand_classes = np.random.choice(list(set(test_classes_pool) - (set(test_classes))), size=n_val)
            self.val_classes_list.append(list(rand_classes))
        self.n_test = n_test
        self.n_val = n_val
        self.n_fold = n_fold
        with open(ZSL_SPLIT_FN, 'w') as f:
            f.write('{}'.format({
                    'test_classes_list': self.test_classes_list,
                    'val_classes_list': self.val_classes_list,
                    'n_test': self.n_test,
                    'n_fold': self.n_fold,
                    'n_val': self.n_val
                })
            )

    def get_mask(self, filtered_classes):
        idx = []
        filtered_classes = set(filtered_classes)
        for i in range(len(self.gt_classes)):
            classes = self.gt_classes[i]
            if len(set(classes).intersection(filtered_classes)) == 0:
                idx.append(i)
        idx = np.where(self.split_mask)[0][idx]
        mask = np.zeros_like(self.split_mask).astype(bool)
        mask[idx] = True
        return mask

    def get_train_mask(self, num_set):
        assert num_set < self.n_fold
        filtered_classes = self.test_classes_list[num_set] + self.val_classes_list[num_set]
        return self.get_mask(filtered_classes)

    def get_val_mask(self, num_set):
        assert num_set < self.n_fold
        train_mask = self.get_train_mask(num_set)
        filtered_classes = self.test_classes_list[num_set]
        val_mask = self.get_mask(filtered_classes)
        return self.split_mask & ~train_mask & val_mask

    def get_test_mask(self, num_set=None):
        assert num_set < self.n_fold
        filtered_classes = self.test_classes_list[num_set]
        train_val_mask = self.get_mask(filtered_classes)
        return self.split_mask & ~train_val_mask

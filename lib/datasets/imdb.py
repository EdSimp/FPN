# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 2
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
      return [PIL.Image.open(self.image_path_at(i)).size[0]
              for i in xrange(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                  '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [ [0**2, 1e5**2],    # all
                        [0**2, 32**2],     # small
                        [32**2, 96**2],    # medium
                        [96**2, 1e5**2],   # large
                        [96**2, 128**2],   # 96-128
                        [128**2, 256**2],  # 128-256
                        [256**2, 512**2],  # 256-512
                        [512**2, 1e5**2],  # 512-inf
                      ]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in xrange(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert(gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes' : boxes,
                'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass


    def cal_area_11point(self, dict_PR = None, no_this_class = []):
        area_class = []
        for key in dict_PR:
            if key in no_this_class:
                continue
            cls_pr = dict_PR[key]
            '''if key == 23 or key == 67:
                print 'cls_pr is {}'.format(cls_pr)'''
            #cls_pr.sort()
            area_iou = []
            ordinate_iou = [[] for i in range(len(cls_pr[0][0]))]
            print 'key is {}'.format(key)
            for j in range(len(cls_pr[0][0])):
                tmp = []
                for i in range(len(cls_pr)):
                    p = cls_pr[i][0][j]
                    r = cls_pr[i][1][j]
                    tmp.append((r, p))
                tmp.sort()
                tmp = np.array(tmp)
                ordinate_iou[j].append(list(tmp[:, 1]))
                ordinate_iou[j].append(list(tmp[:, 0]))
        
            for j in range(len(cls_pr[0][0])):
                area = 0.
                max_precs = [0. for i in range(11)]
                p = ordinate_iou[j][0]
                r = ordinate_iou[j][1]
                print 'p is {}, r is {}'.format(p, r)
                start_idx = len(r) - 1
                for k in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if r[i] < float(k) / 10.:
                            start_idx = i
                            if k > 0:
                                max_precs[k - 1] = max_precs[k]
                            break
                        else:
                            if max_precs[k] < p[i]:
                                max_precs[k] = p[i]
                print 'max_precs is {}'.format(max_precs)
                area = np.array(max_precs).mean()
                area_iou.append(area)
            area_class.append(area_iou)
        area_class = np.array(area_class)
        mAP = area_class.mean(axis = 0)
        return mAP, area_class

    def evaluate_mAP(self, all_boxes=None, thresholds=None,
                     area='all', limit=None):        #yuesongtian
        """Evaluate detection proposal mAP metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                  '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [ [0**2, 1e5**2],    # all
                        [0**2, 32**2],     # small
                        [32**2, 96**2],    # medium
                        [96**2, 1e5**2],   # large
                        [96**2, 128**2],   # 96-128
                        [128**2, 256**2],  # 128-256
                        [256**2, 512**2],  # 256-512
                        [512**2, 1e5**2],  # 512-inf
                      ]
        confidence_thresh = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        #confidence_thresh = [ float(i) / 100. for i in range(1,101,5) ]
        dict_PR = {}
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        #num_pos = 0        #it is used for recall
        AP_of_classes = []
        no_this_class = []  #store the class that never appears in test set
        
        
        for thresh in confidence_thresh:
            conf_thresh = thresh
            no_this_class = []
            for k in xrange(1, self.num_classes):
                #for each class C, calculate the precision C
                num_detected = 0
                gt_overlaps = np.zeros(0)
                num_pos = 0
                true_positive = [0 for i in range(len(thresholds))]
                if k not in dict_PR:
                    dict_PR[k] = []
                for i in xrange(self.num_images):
                    # Checking for max_overlaps == 1 avoids including crowd annotations
                    # (...pretty hacking :/)

                    max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
                    gt_inds = np.where((self.roidb[i]['gt_classes'] == k) &
                                       (max_gt_overlaps == 1))[0]
                    gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
                    gt_areas = self.roidb[i]['seg_areas'][gt_inds]
                    valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                             (gt_areas <= area_range[1]))[0]
                    gt_boxes = gt_boxes[valid_gt_inds, :]
                    num_pos += len(valid_gt_inds)
                    
                    if all_boxes is None:
                        # If candidate_boxes is not supplied, the default is to use the
                        # non-ground-truth boxes from this roidb
                        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
                    else:
                        #boxes = candidate_boxes[i]
                        inds = np.where(all_boxes[k][i][:, 4] > conf_thresh)[0]
                        boxes = all_boxes[k][i][inds, :]

                    if boxes.shape[0] == 0:
                        continue
                    if limit is not None and boxes.shape[0] > limit:
                        boxes = boxes[:limit, :]
                        #the number of objects of class C detected by the model 
                    
                    num_detected = num_detected + boxes.shape[0]

                    overlaps = bbox_overlaps(boxes.astype(np.float),
                                             gt_boxes.astype(np.float))

                    _gt_overlaps = np.zeros((gt_boxes.shape[0]))

                    if overlaps.shape[0] == 1:
                        _gt_overlaps = overlaps
                    elif overlaps.shape[0] > 1:
                        for j in xrange(gt_boxes.shape[0]):
                            # find which proposal box maximally covers each gt box
                            argmax_overlaps = overlaps.argmax(axis=0)
                            # and get the iou amount of coverage for each gt box
                            max_overlaps = overlaps.max(axis=0)
                            # find which gt box is 'best' covered (i.e. 'best' = most iou)
                            gt_ind = max_overlaps.argmax()
                            gt_ovr = max_overlaps.max()
                            if gt_ovr < 0:
                                break
                            assert(gt_ovr >= 0)
                            # find the proposal box that covers the best covered gt box
                            box_ind = argmax_overlaps[gt_ind]
                            # record the iou coverage of this gt box
                            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                            assert(_gt_overlaps[j] == gt_ovr)
                            # mark the proposal box and the gt box as used
                            overlaps[box_ind, :] = -1
                            overlaps[:, gt_ind] = -1
                            #print 'overlaps is internal {}, and there are {} gt'.format(overlaps, gt_boxes.shape[0])
                    # append recorded iou coverage level
                    #gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
                    if thresholds is None:
                        step = 0.05
                        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
                    for m, t in enumerate(thresholds):
                        true_positive[m] = true_positive[m] + (_gt_overlaps >= t).sum()
                    
                
                if num_pos == 0:
                    no_this_class.append(k)
                
                #gt_overlaps = np.sort(gt_overlaps)
                if thresholds is None:
                    step = 0.05
                    thresholds = np.arange(0.5, 0.95 + 1e-5, step)
                precision = np.zeros_like(thresholds)
                recall = np.zeros_like(thresholds)
                # compute recall for each iou threshold
                for i, t in enumerate(thresholds):
                    if num_detected != 0:
                        #precision[i] = (gt_overlaps >= t).sum() / float(num_detected)
                        precision[i] = true_positive[i] / float(num_detected)
                    else:
                        precision[i] = 0.
                    if num_pos != 0:
                        #recall[i] = (gt_overlaps >= t).sum() / float(num_pos)
                        recall[i] = true_positive[i] / float(num_pos)
                    else:
                        recall[i] = 0.

                if k == 12:
                    print true_positive, num_detected, num_pos
                dict_PR[k].append((precision, recall))
                #AP_of_classes.append(AP)
        #print 'dict_PR is {}'.format(dict_PR)
        (mAP, area_class) = self.cal_area_11point(dict_PR = dict_PR, no_this_class = no_this_class)
        # ar = 2 * np.trapz(recalls, thresholds)
        #return {'mAP': mAP, 'AP': AP_of_classes, 'thresholds': thresholds,
        #        'no this class': no_this_class, 'gt_overlaps': gt_overlaps}
        return {'mAP': mAP, 'area_class': area_class} 




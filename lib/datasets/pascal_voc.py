# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
from utils.bbox import bbox_overlaps

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'flaw')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ''#'.jpg'
        self._image_index, self._xml_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        
        image_path = os.path.join(index).strip()
        #image_path = os.path.join(self._data_path, 'JPEGImages',
        #                          index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets/' + self._image_set + '.txt')
        print 'self.num_classes is ',self.num_classes
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        #with open(image_set_file) as f:
        #    image_index = [x.strip() for x in f.readlines()]
        with open(image_set_file) as f:
            image_index = [x.split(' ')[0] for x in f.readlines()]
            print 'dengdengdenngdadi', f.readlines()
            for x in f.readlines():
                print x.split(' ')
            f = open(image_set_file)
            xml_index = [x.split(' ')[-1].rstrip('\n') for x in f.readlines()]
        #image_index = image_index[:200] ####add by cjs
        #xml_index = xml_index[:200] ###add by cjs
        print 'len of image_index is', len(image_index), 'len of xml_index is ',len(xml_index)
        
        return image_index, xml_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        gt_roidb = [self._load_pascal_annotation(index) for index in self._xml_index]
        print 'length of gt_roidb is ', len(gt_roidb)

        #gt_roidb = [self._load_pascal_annotation(index)
        #            for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        '''
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        '''
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            #ss_roidb = self._load_selective_search_roidb(gt_roidb)
            #roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
            roidb = gt_roidb
        else:
            #roidb = self._load_selective_search_roidb(None)
            roidb = self.get_roidb()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations_two', index)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = "/home/ayogg/USB/FPN/voc_result.txt" 
        #filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        #path = os.path.join(
        #    self._devkit_path,
        #    'results',
        #    filename)
        #return path
        return filename

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print 'filename voc is ',filename
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1],dets[k, 0] + 1, dets[k, 1] + 1,dets[k, 2] + 1, dets[k, 3] + 1))
    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations_two',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            'all.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                #os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _write_detect_results_file(self, all_boxes, conf_thresh, iou_thresh, net_name):          #added by yuesongtian
        filename = cfg.ROOT_DIR + '/output/FP_Net_end2end/voc_2007_test/' + str(len(self.classes)) + '_' + net_name + '.txt'
        print 'Writing detection results to {}'.format(filename)
        if not os.path.exists(filename):
            os.system(r'touch %s' % filename)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index): 
                 
                
                f.write('{:s}'.format(index))
                
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    max_gt_overlaps = self.roidb[im_ind]['gt_overlaps'].toarray().max(axis=1)
                    gt_inds = np.where((self.roidb[im_ind]['gt_classes'] == cls_ind) &
                                       (max_gt_overlaps == 1))[0]
                    gt_boxes = self.roidb[im_ind]['boxes'][gt_inds, :]
                    gt_areas = self.roidb[im_ind]['seg_areas'][gt_inds]
                    valid_gt_inds = np.where((gt_areas >= 0**2) &
                                             (gt_areas <= 1e5**2))[0]
                    gt_boxes = gt_boxes[valid_gt_inds, :]
                    if (len(all_boxes[cls_ind][im_ind]) == 0):
                        continue
                    inds = np.where(all_boxes[cls_ind][im_ind][:, 4] > conf_thresh)[0]
                    dets_ = all_boxes[cls_ind][im_ind][inds, :]
                    if dets_ == [] or gt_boxes.shape[0] == 0:
                        continue
                    overlaps = bbox_overlaps(dets_.astype(np.float),
                                             gt_boxes.astype(np.float))

                    if overlaps.shape[0] == 1:
                        #print 'overlaps is ', overlaps, overlaps.shape[0], dets_.shape, gt_boxes.shape
                        argmax_overlaps = overlaps.argmax(axis = 0)
                        max_overlaps = overlaps.max(axis = 0)
                        gt_ind = max_overlaps.argmax()
                        gt_ovr = max_overlaps.max()
                        if gt_ovr >= iou_thresh:
                            box_ind = argmax_overlaps[gt_ind]
                            f.write('   {:.1f}({:.3f}) {:.1f} {:.1f} {:.1f} {:.1f}'.
                                format(cls_ind, dets_[box_ind, -1],
                                       dets_[box_ind, 0] + 1, dets_[box_ind, 1] + 1,
                                       dets_[box_ind, 2] + 1, dets_[box_ind, 3] + 1))   
                        if(gt_boxes.shape[0]==1):
                            f.write('*')
                            f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(gt_boxes[0,0]+1,gt_boxes[0,1]+1,gt_boxes[0,2]+1,gt_boxes[0,3]+1))
                        else:
                            f.write('*')
                            for gt_index in range(gt_boxes.shape[0]):
                                f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.format(gt_boxes[gt_index,0]+1,gt_boxes[gt_index,1]+1,gt_boxes[gt_index,2]+1,gt_boxes[gt_index,3]+1))

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
                            box_ind = argmax_overlaps[gt_ind]
                            if gt_ovr < iou_thresh:
                                overlaps[box_ind, :] = -1
                                overlaps[:, gt_ind] = -1
                                continue
                            # write box > iou_thresh to f
                            f.write('   {:.1f}({:.3f}) {:.1f} {:.1f} {:.1f} {:.1f}'.
                                format(cls_ind, dets_[box_ind, -1],
                                       dets_[box_ind, 0] + 1, dets_[box_ind, 1] + 1,
                                       dets_[box_ind, 2] + 1, dets_[box_ind, 3] + 1))
                            # mark the proposal box and the gt box as used
                            overlaps[box_ind, :] = -1
                            overlaps[:, gt_ind] = -1 

                        if(gt_boxes.shape[0]==1):
                            f.write('*')
                            f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(gt_boxes[0,0]+1,gt_boxes[0,1]+1,gt_boxes[0,2]+1,gt_boxes[0,3]+1))
                        else:
                            f.write('*')
                            for gt_index in range(gt_boxes.shape[0]):
                                f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(gt_boxes[gt_index,0]+1,gt_boxes[gt_index,1]+1,gt_boxes[gt_index,2]+1,gt_boxes[gt_index,3]+1))

                    else:
                        if(gt_boxes.shape[0]==1):
                            f.write('*')
                            f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(gt_boxes[0,0]+1,gt_boxes[0,1]+1,gt_boxes[0,2]+1,gt_boxes[0,3]+1))
                        else:
                            f.write('*')
                            for gt_index in range(gt_boxes.shape[0]):
                                f.write('   {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(gt_boxes[gt_index,0]+1,gt_boxes[gt_index,1]+1,gt_boxes[gt_index,2]+1,gt_boxes[gt_index,3]+1))
                f.write('\n')                
        f.close()
        return filename


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()

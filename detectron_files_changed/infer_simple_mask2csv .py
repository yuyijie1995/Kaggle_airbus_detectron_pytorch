from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange
import pycocotools.mask as mask_util

import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


class Airbus_Submit:
    def __init__(self,thresh=0.4,csv_file_name='rle1.csv'):
        self.thresh=thresh
        self.csv_file_name=csv_file_name

        self.csv_img=[]#save img name
        self.csv_rle=[]#save rle result
        self.csv_con=[]#save confidence
        self.csv_area=[]#save the area of mask
    def convert_from_cls_format(self,cls_boxes,cls_segms,cls_keyps):
        box_list=[b for b in cls_boxes if len(b)>0]
        if len(box_list)>0:
            boxes=np.concatenate(box_list)
        else:
            boxes=None
        if cls_segms is not None:
            segms=[s for slist in cls_segms for s in slist]
        else:
            segms=None

        if cls_keyps is not None:
            keyps=[k for klist in cls_keyps for k in klist]
        else:
            keyps=None
        classes=[]

        for j in range(len(cls_boxes)):
            classes+=[j]*len(cls_boxes)
        return boxes,segms,keyps,classes

    def rle_encode(self,img):
        pixels=img.T.flatten()
        pixels=np.concatenate([[0],pixels,[0]])
#        run1=np.where(pixels[1:]!=pixels[:-1])
        # 只有条件（condition），没有x和y，则输出满足条件（即非0）元素的坐标，这里的坐标用tuple的形式给出，通常原数组有多少维，输出
        #的tuple中就包含几个数组，分别对应符合条件元素的各维坐标
        runs=np.where(pixels[1:]!=pixels[:-1])[0]+1
        runs[1::2]-=runs[::2]
        return ' '.join(str(x) for x in runs)


    def extract_result(self,cls_boxes,cls_segms,cls_keyps,im_real_name,confidence):
        if isinstance(cls_boxes,list):
            boxes,segms,keypoints,classes=self.convert_from_cls_format(cls_boxes,cls_segms,cls_keyps)

        if (boxes is None or boxes.shape[0]==0 or max(boxes[:,4])<self.thresh):
            return

        if segms is not None and len(segms)>0:
            masks=np.array(mask_util.decode(segms))

        if masks is None:# 这意味着在图片中，所有目标的置信度都小于阈值
            return

        self.mask_to_rle_csv(im_real_name,masks,confidence)

    def mask_to_rle_csv(self,img,masks,confidence):
        index=np.argsort(-confidence)
        bg=np.zeros((768,768),dtype=np.uint8)
        for i in index:
            mask=masks[:,:,i]
            if (mask is None or confidence[i]<self.thresh):
                continue
            mask_xor=(mask^bg)&mask
            #使用异或^的思路，因为每一个目标都不能有重叠的部分，按置信度排序后，第二个目标与第一个目标做异或，然后再和自己做一个
            #与操作得到一个被第一个目标挖去共有部分的mask
            area=mask_xor.sum()
            if (area==0):
                continue
            print(confidence[i])
            rle=self.rle_encode(mask_xor)
            bg+=mask_xor

            self.csv_img.append(img)
            self.csv_rle.append(rle)
            self.csv_con.append(confidence[i])
            self.csv_area.append(area)

    def create_csv(self):
        df=pd.DataFrame({'ImageId':self.csv_img,'EncodedPixels':self.csv_rle,'confidence':self.csv_con,'area':self.csv_area})
        #df=df[['ImageId','EncodedPixels','confidence','area']]
        df.to_csv(self.csv_file_name,index=False,sep=str(','))
        print('%s is written successfully.'%self.csv_file_name)





def main():
    """main function"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    airbus = Airbus_Submit(thresh=0.7, csv_file_name='../rle.csv')

    for i in xrange(num_images):
        print('img', i)
        im_name=imglist[i]
        im = cv2.imread(imglist[i])
        assert im is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
        airbus.extract_result(cls_boxes, cls_segms, cls_keyps,
                              im_real_name=im_name.split('/')[-1], confidence=cls_boxes[1][:, 4])
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
    airbus.create_csv()

    if args.merge_pdfs and num_images > 1:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

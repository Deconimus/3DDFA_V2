# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    
    print(ver_lst[0].shape)
    
    print(tddfa.bfm.u.shape)
    
    lm68 = np.reshape(np.reshape(ver_lst[0].T, (-1,1))[tddfa.bfm.keypoints], (-1,3))
    print(lm68.shape)
    
    for i in range(lm68.shape[0]):
        lm68[i,1] = img.shape[0] - lm68[i,1]
    
    for i in range(ver_lst[0].shape[1]):
        ver_lst[0][1,i] = img.shape[0] - ver_lst[0][1,i]
        
    useful_tri = np.copy(tddfa.tri)
        
    for i in range(useful_tri.shape[0]):
        tmp = useful_tri[i,2]
        useful_tri[i,2] = useful_tri[i,0]
        useful_tri[i,0] = tmp
        
    useful_tri = useful_tri + 1
    
    np.save("asd_lm.npy", lm68)
    np.save("asd_v.npy", ver_lst[0].T)
    np.save("asd_f.npy", useful_tri)
    
    
    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)

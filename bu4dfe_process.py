# coding: utf-8

__author__ = 'cleardusk'

import numpy as np
import sys, argparse, cv2, yaml, pathlib, _pickle, pickle, os

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
    
    # input and output folder
    image_path = "G:\\BU-4DFE\\Extracted"
    save_path  = "G:\\bu4dfe_3ddfa_proc"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()
    
    img_list = sorted([str(x) for x in pathlib.Path(image_path).rglob("*.jpg")])
    
    print("Reconstructing:\n")
    
    for file in img_list:
        
        out_file = save_path + file[len(image_path):-4] + ".pickle"
        if os.path.isfile(out_file):
            continue
            
        out_dir = str(pathlib.Path(out_file).parent)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    
        # Given a still image path and load to BGR channel
        img = cv2.imread(file)

        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, skipping \"'+file+'\".')
            continue
        #print(f'Detect {n} faces')

        param_lst, roi_box_lst = tddfa(img, boxes)

        # Visualization and serialization
        dense_flag = True

        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        # repair all the bs because of the matrices' axis alignment...
        
        lm68 = np.reshape(np.reshape(ver_lst[0].T, (-1,1))[tddfa.bfm.keypoints], (-1,3))
        
        for i in range(lm68.shape[0]):
            lm68[i,1] = img.shape[0] - lm68[i,1]
        
        for i in range(ver_lst[0].shape[1]):
            ver_lst[0][1,i] = img.shape[0] - ver_lst[0][1,i]
            
        vertices = ver_lst[0].T
            
        useful_tri = np.copy(tddfa.tri)
            
        for i in range(useful_tri.shape[0]):
            tmp = useful_tri[i,2]
            useful_tri[i,2] = useful_tri[i,0]
            useful_tri[i,0] = tmp
            
        #useful_tri = useful_tri + 1
        
        # save
        
        mesh = dict()
        mesh["vertices"] = vertices
        mesh["faces"] = useful_tri
        mesh["lm68"] = lm68
        
        with open(out_file, "wb+") as f:
            _pickle.dump(mesh, f)
            
        print(out_file)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a mesh')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str)
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')

    args = parser.parse_args()
    main(args)

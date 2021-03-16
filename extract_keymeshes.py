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

import lib.wavefront


bu4_path = "G:\\BU-4DFE\\Extracted"
bu4_save_path = "D:\\bu4_key_rec"
fs_path = "E:\\mv_selection"
fs_save_path = "D:\\fs_key_rec"


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()
    
    dataset = args.dataset.lower().strip()
    if dataset.startswith("f"):
        dataset = "facescape"
    else:
        dataset = "bu4dfe"
    
    files = []
    
    if dataset == "facescape":
        
        save_path = fs_save_path
        data_path = fs_path
        
        files = sorted([str(f) for f in pathlib.Path(data_path).rglob("*_0.jpg") if int(f.parent.name) >= 344])
        
    else:
        
        save_path = bu4_save_path
        data_path = bu4_path
        
        keynums = [25, 70]
        
        for n in keynums:
            files += [str(f) for f in pathlib.Path(data_path).rglob("*"+str(n).zfill(3)+".jpg")]
        files += [str(f)+"/000.jpg" for f in pathlib.Path(data_path).rglob("Angry") if os.path.isdir(str(f))]
        files = sorted(files)
    
    for file in files:
        
        out_file = save_path + os.path.sep + "3ddfav2" + file[len(data_path):-4] + ".obj"
        out_dir = str(pathlib.Path(out_file).parent)
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        elif os.path.isfile(out_file):
            print("Skipping file: \""+file+"\".")
            continue
    
        # Given a still image path and load to BGR channel
        img = cv2.imread(file)

        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            continue
        #print(f'Detect {n} faces')

        param_lst, roi_box_lst = tddfa(img, boxes)

        # Visualization and serialization
        dense_flag = True

        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        # repair all the bs because the matrices axis alignment...
        
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
        
        if False:
        
            mesh = dict()
            mesh["vertices"] = vertices
            mesh["faces"] = useful_tri
            mesh["lm68"] = lm68
            
            out_file = file[:-4]+".pickle"
            
            with open(out_file, "wb+") as f:
                _pickle.dump(mesh, f)
                
        else:
            
            obj = lib.wavefront.Wavefront()
            obj.vertices = vertices
            obj.facevertices = useful_tri+1
            
            obj.write(out_file)
            
            print(out_file)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a mesh')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', type=str, default='gpu', help='gpu or cpu mode')
    parser.add_argument('-d', '--dataset', type=str, default='facescape', help='facescape or bu4dfe')

    args = parser.parse_args()
    main(args)

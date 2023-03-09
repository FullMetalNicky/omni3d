# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
import pandas as pd
import open3d as o3d
import copy
from Omni3DDataset import *

# python3 demo/vomit.py --config configs/Base_Omni3D.yaml --input-folder "datasets/Omni3D/Hypersim_test.json" --threshold 0.25 MODEL.WEIGHTS trained_models/22_10_28_server_224/model_recent.pth 

def processCube(origin, pcd):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    pcd_t = copy.deepcopy(pcd).transform(T)

    return pcd_t


def do_test(args, cfg, model):
    annotations = []

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

    with open(args.input_folder) as f:
        js = json.load(f)

    images = js['images']
    list_of_ims = []
    image_ids = []
    seq_ids = []
    
    image_objects = []
    category_objects = []

    for image in images:
        list_of_ims.append(os.path.join('datasets', image['file_path']))
        image_ids.append(image['id'])
        seq_ids.append(image['seq_id'])
        img_obj = Omni3DImage(image['id'], image['dataset_id'], image['seq_id'], image['width'], image['height'], image['file_path'], image['K'], image['gt'], image['t'])
        image_objects.append(img_obj)

    cats = js['categories']
    for cat in cats:
        cat_obj = Omni3DCategory(cat['id'], cat['name'])
        category_objects.append(cat_obj)

    model.eval()

    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    count = 0
            
    for frame, path in enumerate(list_of_ims):

        im_name = util.file_parts(path)[1]
        im = util.imread(path)

        if im is None:
            continue
        
        image_shape = im.shape[:2]  # h, w

        h, w = image_shape
        f_ndc = 4
        f = f_ndc * h / 2

        K = np.array([
            [380.07025146484375, 0.0, 324.4729919433594], #[f, 0.0, w/2], 
           [0.0, 379.66119384765625, 237.78517150878906], #[0.0, f, h/2], 
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        dets = model(batched)[0]['instances']
        n_det = len(dets)
        meshes = []
        meshes_text = []

        detections = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):
                
                # skip
                if score < thres:
                    continue
                
                cat = cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
                print('File: {} with {} dets'.format(im_name, len(meshes)))

                center_cam = center_cam.tolist()
                dimensions = dimensions.tolist()
                pose = pose.tolist()
                cat_idx = cat_idx.cpu().detach().numpy()
                score = score.cpu().detach().numpy().item()
                
                detection = dets[idx]
                box = detection.pred_boxes.tensor.cpu().detach().tolist()[0]
                box3D = detection.pred_bbox3D.cpu().detach().tolist()[0]
                category = detection.pred_classes.cpu().detach().item()
                pose = detection.pred_pose.cpu().detach().tolist()[0]
                #print(category_objects[category].name)
                obj = Omni3DObject(dataset_id=11, 
                                   image_id=image_ids[frame],
                                   seq_id=seq_ids[frame],
                                   category_id=category,
                                   category_name=category_objects[category].name,
                                   valid3D=1,
                                   bbox2D_tight=box,
                                   bbox2D_proj=box,
                                   bbox2D_trunc=box,
                                   bbox3D_cam=box3D,
                                   center_cam=center_cam,
                                   dimensions=dimensions,
                                   score=score,
                                   R_cam=pose)
                annotations.append(obj)

    info = Omni3DInfo(js['info']['id'], js['info']['source'], js['info']['name'], js['info']['split'] , js['info']['version'], js['info']['url'])
    dataset = Omni3DDataset(info=info, images=image_objects, categories=category_objects, annotations=annotations)
    with open("predictions.json", "w") as f:
        f.write("{}".format(dataset.toJSON()))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    # import ipdb
    # ipdb.set_trace()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

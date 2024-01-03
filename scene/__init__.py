#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from imgviz import label_colormap
import torch
from scene.replica_datasets import ReplicaDatasetCache
import numpy as np
from utils.image_utils import plot_semantic_legend

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif "Replica" in args.source_path:
            print("----- Replica Dataset -----")
            total_num = 900
            step = 5
            train_ids = list(range(0, total_num, step))
            test_ids = [x+step//2 for x in train_ids] 
            replica_data_loader = ReplicaDatasetCache(data_dir=args.source_path, train_ids=train_ids, test_ids=test_ids)
            print("--------------------")
            if args.sparse_views:
                if len(args.label_map_ids)>0:
                    print("Use label maps only for selected frames, ", args.label_map_ids)
                    replica_data_loader.sample_specific_labels(args.label_map_ids, train_ids)
                else:
                    print("Sparse Labels Mode! Sparsity Ratio is ", args.sparse_ratio)
                    replica_data_loader.sample_label_maps(sparse_ratio=args.sparse_ratio, random_sample=args.random_sample, load_saved=args.load_saved)
            else:
                print("Standard setup with full dense supervision.")
            scene_info = sceneLoadTypeCallbacks["Replica"](args, replica_data_loader)
            
            # preprocess semantic info
            self.semantic_classes = torch.from_numpy(replica_data_loader.semantic_classes)
            self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including void class=0
            self.num_valid_semantic_class = self.num_semantic_class - 1  # exclude void class
            assert self.num_semantic_class==replica_data_loader.num_semantic_class

            json_class_mapping = os.path.join(args.source_path, "info_semantic.json")
            with open(json_class_mapping, "r") as f:
                annotations = json.load(f)
            instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
            total_num_classes = len(annotations["classes"])
            assert total_num_classes==101  # excluding void we have 102 classes
            # assert self.num_valid_semantic_class == np.sum(np.unique(instance_id_to_semantic_label_id) >=0 )

            colour_map_np = label_colormap(total_num_classes)[replica_data_loader.semantic_classes] # select the existing class from total colour map
            self.colour_map = torch.from_numpy(colour_map_np)
            self.valid_colour_map  = torch.from_numpy(colour_map_np[1:,:]) # exclude the first colour map to colourise rendered segmentation without void index

            # plot semantic label legend
            # class_name_string = ["voild"] + [x["name"] for x in annotations["classes"] if x["id"] in np.unique(data.semantic)]
            class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
            plot_semantic_legend(replica_data_loader.semantic_classes, class_name_string, 
                                colormap=label_colormap(total_num_classes+1), save_path=args.model_path)
            # total_num_classes +1 to include void class            
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # print("Generating Video Cameras")
            # self.video_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.video_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            print(self.gaussians.get_xyz.shape[0])
            sys.exit(0)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.num_valid_semantic_class)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    # def getVideoCameras(self, scale=1.0):
    #     return self.video_cameras[scale]
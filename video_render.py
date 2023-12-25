import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import imageio
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
import sys
# import dinov2.eval.segmentation_m2f.models.segmentors
# import dinov2.eval.segmentation.utils.colormaps as colormaps
# import math
# import itertools
# from functools import partial
# import urllib
# import mmcv
# from mmcv.runner import load_checkpoint
# import torch.nn.functional as F
# from mmseg.apis import init_segmentor, inference_segmentor
# import dinov2.eval.segmentation.utils.colormaps as colormaps
# import dinov2.eval.segmentation.models
from argparse import ArgumentParser
import cv2, imageio, time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# def load_config_from_url(url: str) -> str:
#     with urllib.request.urlopen(url) as f:
#         return f.read().decode()

# def render_segmentation(segmentation_logits, dataset):
#     DATASET_COLORMAPS = {
#         "ade20k": colormaps.ADE20K_COLORMAP,
#         "voc2012": colormaps.VOC2012_COLORMAP,
#     }
#     colormap = DATASET_COLORMAPS[dataset]
#     colormap_array = np.array(colormap, dtype=np.uint8)
#     segmentation_values = colormap_array[segmentation_logits + 1]
#     # return Image.fromarray(segmentation_values)
#     return segmentation_values

def render_video(dataset, iteration, pipeline, seg_model, mask_generator=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    source_path = dataset.source_path
    views = scene.getTrainCameras()
    view = views[0]
    
    # render_poses = generate_spiral_path(source_path, n_frames=90, n_rots=2)
    if dataset.source_path.find('llff') != -1:
        render_poses = generate_spiral_path(source_path, n_frames=90, n_rots=2)
    elif dataset.source_path.find('360') != -1:
        render_poses = generate_ellipse_path(views, n_frames=90)
    
    if not os.path.exists(os.path.join(dataset.model_path, 'video_frames')):
        os.makedirs(os.path.join(dataset.model_path, 'video_frames'))
          
    rgbs = []
    masked_imgs = []
    fps = []
    for idx, pose in enumerate(tqdm(render_poses, desc="Video Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        
        start = time.time()
        rgb = render(view, gaussians, pipeline, background)["render"]
        rgb = torchvision.transforms.Resize((378, 504))(rgb)
        # rgb = cv2.resize(rgb.detach().cpu().numpy().transpose([1, 2, 0]), (1008, 756))
        # cv2.imwrite(os.path.join(dataset.model_path, 'video_frames', '{0:03d}'.format(idx) + ".png"), rgb)
        torchvision.utils.save_image(rgb, os.path.join(dataset.model_path, 'video_frames', '{0:03d}'.format(idx) + ".png"))
        rgb = to8b(rgb.detach().cpu().numpy()).transpose(1, 2, 0)
        
        # # segment
        # array = np.array(rgb)[:, :, ::-1] # BGR
        # segments = inference_segmentor(model, array)
        # end = time.time()
        # fps.append(end-start)
        # segmentation_logits = segments[0]
        # segmentation_values = render_segmentation(segmentation_logits, "ade20k")
        # masked_img = (1 - args.alpha) * rgb + args.alpha * segmentation_values
        # masked_img = masked_img.astype(np.uint8)
        # masked_imgs.append(masked_img)
        rgbs.append(rgb)
        
    rgbs = np.stack(rgbs, 0)
    # masked_imgs = np.stack(masked_imgs, 0)
    imageio.mimwrite(os.path.join(dataset.model_path, 'render.mp4'), rgbs, fps=30, quality=8)
    # imageio.mimwrite(os.path.join(dataset.model_path, 'seg.mp4'), to8b(masked_imgs), fps=30, quality=8)
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--spherify", action="strore_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # #! Load pretrained segmentation model (Mask2Former)
    # print("Load pretrained segmentation model (Mask2Former)......")
    # DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    # CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
    # CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

    # cfg_str = load_config_from_url(CONFIG_URL)
    # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    # # cfg = mmcv.Config.fromfile('/data1/shengxiangji/dinov2/configs/dinov2_vitg14_ade20k_m2f_config.py')
    
    # seg_model = init_segmentor(cfg)
    # load_checkpoint(model, "/data1/shengxiangji/dinov2/checkpoints/dinov2_vitg14_ade20k_m2f.pth", map_location="cpu")
    # # checkpoint = torch.load("/data1/shengxiangji/dinov2/checkpoints/dinov2_vitg14_ade20k_m2f.pth")
    # # model.load_state_dict(checkpoint)
    # seg_model.cuda()
    # seg_model.eval()
    
    
    with torch.no_grad():
        render_video(model.extract(args), args.iteration,  pipeline.extract(args), None)
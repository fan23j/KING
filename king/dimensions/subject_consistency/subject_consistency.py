import argparse
import os
import os.path as osp
import time
import cv2
import torch
import json

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp.yolox_x_sportsmot_mix import Exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.byte_tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from king.utils import load_video
import logging

from king.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from ..cache import cache_dimensions

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class ByteTrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.fps = 8
        self.save_result = True

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=True
    ):
        self.model = model
        self.device = device
        self.test_size = exp.test_size
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.num_classes = exp.num_classes

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def imageflow_demo(predictor, video_list, vis_folder, test_size):
    args = ByteTrackerArgs()
    video_results = {}
    video_tracking_results = {} # store tracking results for each video, save to cache
    for video_info in video_list:
        video_path = video_info['video_list'][0]
        video_name = osp.basename(video_path)
        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        save_path = osp.join(save_folder, video_path.split("/")[-1])
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(width), int(height))
        )
        tracker = BYTETracker(args, frame_rate=fps)
        timer = Timer()
        frame_id = 0
        results = []
        while True:
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret_val, frame = cap.read()
            if ret_val:
                outputs, img_info = predictor.inference(frame, timer)
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )
                else:
                    timer.toc()
                    online_im = img_info['raw_img']
                if args.save_result:
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}_{video_path.split('/')[-1]}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

        # Compute the consistency metric for this video
        consistency_score = compute_consistency_metric(results, total_frames)
        video_results[video_path] = consistency_score
        video_tracking_results[video_name] = results
        logger.info(f"Consistency score for {video_path}: {consistency_score:.4f}")

    # Compute the overall consistency score
    if video_results:
        all_results = sum(video_results.values()) / len(video_results)
    else:
        all_results = 0.0
    return all_results, video_results, video_tracking_results

def compute_consistency_metric(results, total_frames):
    """
    Compute the consistency metric for a single video.
    Args:
        results (list): List of tracking results as strings.
        total_frames (int): Total number of frames in the video.
    Returns:
        float: Consistency score for the video.
    """
    track_frames = {}
    for line in results:
        frame_id, tid, x1, y1, w, h, score, _, _, _ = line.strip().split(',')
        frame_id = int(frame_id)
        tid = int(tid)
        if tid not in track_frames:
            track_frames[tid] = []
        track_frames[tid].append(frame_id)

    per_track_consistency = []
    for tid, frames in track_frames.items():
        frames = sorted(frames)
        max_continuous_length = 1
        current_length = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                current_length += 1
            else:
                if current_length > max_continuous_length:
                    max_continuous_length = current_length
                current_length = 1
        if current_length > max_continuous_length:
            max_continuous_length = current_length

        consistency_i = max_continuous_length / total_frames
        per_track_consistency.append(consistency_i)

    if per_track_consistency:
        consistency_score = sum(per_track_consistency) / len(per_track_consistency)
    else:
        consistency_score = 0.0

    return consistency_score

@cache_dimensions
def eval_subject_consistency(json_dir, device, submodules_list, **kwargs):
    with open(json_dir, "r") as f:
        video_list = json.load(f)
    video_list = distribute_list_to_rank(video_list)
    exp = Exp()
    model = exp.get_model().to(device)
    model.eval()
    
    ckpt_file = submodules_list[0]
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    model = model.half()  # to FP16

    predictor = Predictor(model, exp, device)
    vis_folder = "./vis_results"
    all_results, video_results, video_tracking_results = imageflow_demo(predictor, video_list, vis_folder, exp.test_size)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results, video_tracking_results
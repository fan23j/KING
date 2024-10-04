import argparse
import os
import os.path as osp
import time
import cv2
import torch
import json
from torchvision.ops import box_iou

from loguru import logger
import logging

from king.distributed import (
    get_world_size,
    gather_list_of_dict,
)
from ..cache import DimensionsCache

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_miou_metric(cached_tracking_results, gt_bbox_list_path):
    """
    Compute the mIOU metric for the tracking results.
    Args:
        cached_tracking_results (Dict): Tracking results for the videos.
    Returns:
        float: Overall mIOU for all videos.
        Dict: mIOU for each video.
    """
    video_results = []
    total_iou = 0.0
    total_boxes = 0
    for video_name, tracking_results in cached_tracking_results.items():
        gt_path = os.path.join(gt_bbox_list_path, video_name.replace('.mp4', '.pth'))
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth for video {video_name} not found at path {gt_path}.")
            continue

        ground_truth_boxes = torch.load(gt_path)  # Shape: [num_frames, num_objects, 4]

        # Scale the ground truth boxes: multiply x-coordinates by 720, y-coordinates by 480
        ground_truth_boxes_scaled = ground_truth_boxes.clone()
        ground_truth_boxes_scaled[..., [0, 2]] *= 720  # x1 and x2
        ground_truth_boxes_scaled[..., [1, 3]] *= 480  # y1 and y2

        # Process tracking results
        tracking_boxes_per_frame = {}  # Dict[frame_id, List[boxes]]

        for line in tracking_results:
            frame_id, tid, x1, y1, w, h, score, _, _, _ = line.strip().split(',')
            frame_id = int(frame_id)
            x1 = float(x1)
            y1 = float(y1)
            w = float(w)
            h = float(h)
            x2 = x1 + w
            y2 = y1 + h
            box = [x1, y1, x2, y2]
            if frame_id not in tracking_boxes_per_frame:
                tracking_boxes_per_frame[frame_id] = []
            tracking_boxes_per_frame[frame_id].append(box)

        # Compute IoU per frame
        per_frame_ious = []
        num_frames = ground_truth_boxes_scaled.shape[0]
        for frame_idx in range(num_frames):  # For each frame
            gt_boxes = ground_truth_boxes_scaled[frame_idx]  # Shape: [num_objects, 4]
            gt_boxes_list = gt_boxes.tolist()
            # Remove invalid gt boxes (e.g., zeros)
            gt_boxes_list = [box for box in gt_boxes_list if any(coord != 0 for coord in box)]

            if not gt_boxes_list:
                logger.debug(f"No ground truth boxes in frame {frame_idx} for video {video_name}")
                continue  # Skip frames with no ground truth boxes

            # Get predicted boxes for this frame
            pred_boxes = tracking_boxes_per_frame.get(frame_idx, [])
            if not pred_boxes:
                logger.debug(f"No predicted boxes in frame {frame_idx} for video {video_name}")
                continue  # No predictions for this frame

            # Convert to tensors
            pred_boxes_tensor = torch.tensor(pred_boxes)  # Shape: [num_pred_boxes, 4]
            gt_boxes_tensor = torch.tensor(gt_boxes_list)  # Shape: [num_gt_boxes, 4]

            logger.debug(f"Frame {frame_idx}, video {video_name}, num_pred_boxes: {len(pred_boxes)}, num_gt_boxes: {len(gt_boxes_list)}")
            logger.debug(f"Predicted boxes: {pred_boxes}")
            logger.debug(f"Ground truth boxes: {gt_boxes_list}")

            # Compute IoUs between pred_boxes and gt_boxes
            ious = box_iou(pred_boxes_tensor, gt_boxes_tensor)  # Shape: [num_pred_boxes, num_gt_boxes]

            # For each pred_box, get the max IoU
            max_ious, _ = ious.max(dim=1)  # Shape: [num_pred_boxes]
            per_frame_ious.extend(max_ious.tolist())

            logger.debug(f"Frame {frame_idx}, video {video_name}, per-frame IoUs: {max_ious.tolist()}")

        # Compute mean IoU for this video
        if per_frame_ious:
            mean_iou = sum(per_frame_ious) / len(per_frame_ious)
        else:
            mean_iou = 0.0
        video_results.append({'video_path': video_name, 'video_results': mean_iou})
        total_iou += sum(per_frame_ious)
        total_boxes += len(per_frame_ious)

        logger.info(f"Computed mIOU for video {video_name}: {mean_iou}")

    # Compute overall mIOU
    if total_boxes > 0:
        all_results = total_iou / total_boxes
    else:
        all_results = 0.0

    logger.info(f"Computed overall mIOU: {all_results}")

    return all_results, video_results
    

def eval_miou(json_dir, device, submodules_list, **kwargs):
    cache = DimensionsCache()
    if not cache.has(json_dir):
        raise Exception("mIOU requires tracking results to be cached. Please run dimension `subject_consistency` first.")
    cached_tracking_results = cache.get(json_dir)[2]
    gt_bbox_list_path = '/mnt/bum/hanyi/data/gt_bbox'
    #     video_path_list.append(cache.get(json_dir)[1][i]['video_path'])
    all_results, video_results = compute_miou_metric(cached_tracking_results, gt_bbox_list_path)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
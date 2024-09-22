import argparse
import os
import os.path as osp
import time
import cv2
import torch
import json
from torchvision.ops import box_iou
from torchvision import transforms
import numpy as np

from loguru import logger
import logging
from transformers import TimesformerModel, TimesformerConfig, AutoImageProcessor
import imageio
from king.distributed import (
    get_world_size,
    gather_list_of_dict,
    distribute_list_to_rank,
)
from ..cache import DimensionsCache
import av

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: standardize scale?
SCALE_FACTOR = 100

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def compute_statistics(features):
    mu = torch.mean(features, dim=0)
    sigma = torch_cov(features, rowvar=False)
    return mu, sigma

def torch_cov(tensor, rowvar=True):
    """
    Estimate a covariance matrix (np.cov equivalent) given data.
    """
    if tensor.dim() > 2:
        raise ValueError("tensor has more than 2 dimensions")
    if tensor.dim() < 2:
        tensor = tensor.view(1, -1)
    if not rowvar and tensor.size(0) != 1:
        tensor = tensor.t()
    mean = torch.mean(tensor, dim=1, keepdim=True)
    tensor = tensor - mean
    cov = tensor.matmul(tensor.t()) / (tensor.size(1) - 1)
    return cov

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Fréchet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2

    # Product of covariance matrices
    covmean, _ = sqrtm_newton_schulz(sigma1 @ sigma2)

    if not torch.isfinite(covmean).all():
        logger.warning("FID calculation produces singular product; adding epsilon to diagonal of covariances")
        offset = torch.eye(sigma1.size(0), device=sigma1.device) * eps
        covmean, _ = sqrtm_newton_schulz((sigma1 + offset) @ (sigma2 + offset))

    tr_covmean = torch.trace(covmean)

    fd = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return fd * SCALE_FACTOR

def sqrtm_newton_schulz(A, numIters=50):
    """Compute matrix square root using the Newton-Schulz method."""
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.norm()
    Y = A / normA
    I = torch.eye(dim, device=A.device).expand_as(A)
    Z = torch.eye(dim, device=A.device).expand_as(A)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    sA = Y * torch.sqrt(normA)
    return sA, None

def compute_fvd_vae_metric(model, image_processor, video_list, device):
    """
    Compute the FVD metric for the VAE model.
    Args:
        model (torch.nn.Module): VAE model.
        video_list (List): List of videos.
        device (torch.device): Device to run the computation on.
    Returns:
        float: Overall FVD for all videos.
        Dict: FVD for each video.
    """
    video_results = []
    total_fvd = 0.0
    total_videos = 0

    # for root, _, files in os.walk('/mnt/mir/fan23j/data/eval-conditions'):
    #     for file in files:
    #         if file.endswith('_video.pt'):
    #             vid = torch.load(os.path.join(root, file))
    #             inputs = image_processor(vid, return_tensors="pt").to(device)
    #             with torch.no_grad():
    #                 encoded_frames = model(**inputs).last_hidden_state.cpu()
    #             torch.save(encoded_frames, os.path.join('king/dimensions/fvd_classifier/groundtrth', file))

    # import pudb; pudb
    for video_info in video_list:
        video_path = video_info['video_list'][0]
        video_name = osp.basename(video_path).split('.')[0]
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=48, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

        inputs = image_processor(list(video), return_tensors="pt").to(device)
        with torch.no_grad():
            encoded_frames = model(**inputs).last_hidden_state.cpu() # [1, 9409, 768]

        # Load ground truth
        gt_path = osp.join('king/dimensions/fvd_vae/groundtruth', f'{video_name}.pt')
        if not osp.exists(gt_path):
            logger.warning(f"Ground truth for video {video_name} not found at path {gt_path}.")
            continue
        
        gt_encoded_frames = torch.load(gt_path, map_location=device).to(torch.float32)

        # Timesformer outputs already flattened
        gen_features_flat = encoded_frames.squeeze(0)
        gt_features_flat = gt_encoded_frames.squeeze(0)

        # Compute statistics
        mu_gen, sigma_gen = compute_statistics(gen_features_flat)
        mu_real, sigma_real = compute_statistics(gt_features_flat)

        # Compute FVD
        fvd = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        logger.info(f"Video: {video_path}, FVD: {fvd.item()}")
        video_results.append({'video_path': video_path, 'video_results': fvd.item()})
        total_fvd += fvd.item()
        total_videos += 1

    if total_videos:
        all_results = total_fvd / total_videos
    else:
        all_results = 0.0
    logger.info(f"Overall FVD Classifier: {all_results}, Scale factor: {SCALE_FACTOR}")
    return all_results, video_results

def eval_fvd_classifier(json_dir, device, submodules_list, **kwargs):
    with open(json_dir, "r") as f:
        video_list = json.load(f)
    video_list = distribute_list_to_rank(video_list)
    ckpt_file = submodules_list['model_path']

    config = TimesformerConfig(image_size=224, num_frames=42, attention_type='divided_space_time')

    model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    all_results, video_results = compute_fvd_vae_metric(model, image_processor, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
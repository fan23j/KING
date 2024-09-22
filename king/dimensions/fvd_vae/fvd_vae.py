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
from diffusers import AutoencoderKLCogVideoX
import imageio
from king.distributed import (
    get_world_size,
    gather_list_of_dict,
    distribute_list_to_rank,
)
from ..cache import DimensionsCache

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: standardize scale?
SCALE_FACTOR = 100

def flatten_features(features):
    if features.dim() == 5:
        batch_size, D, C, H, W = features.shape
        features = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
    else:
        raise ValueError("Features tensor has unexpected shape.")
    return features

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

def compute_fvd_vae_metric(model, video_list, device):
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

    for video_info in video_list:
        video_path = video_info['video_list'][0]
        video_name = osp.basename(video_path).split('.')[0]
        video_reader = imageio.get_reader(video_path, "ffmpeg")

        frames = [transforms.ToTensor()(frame) for frame in video_reader]
        video_reader.close()
        total_frames = len(frames)
        assert total_frames == 49, f"Video {video_path} has {total_frames} frames, expected 49 frames."

        frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(torch.bfloat16)
        
        # Encode samples
        with torch.no_grad():
            encoded_frames = model.encode(frames_tensor)[0].sample().to(torch.float32)

        # Load ground truth
        gt_path = osp.join('king/dimensions/fvd_vae/groundtruth', f'{video_name}.pt')
        if not osp.exists(gt_path):
            logger.warning(f"Ground truth for video {video_name} not found at path {gt_path}.")
            continue
        
        gt_encoded_frames = torch.load(gt_path, map_location=device).to(torch.float32)

        # Flatten features
        gen_features_flat = flatten_features(encoded_frames)
        gt_features_flat = flatten_features(gt_encoded_frames)

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
    logger.info(f"Overall FVD VAE: {all_results}, Scale factor: {SCALE_FACTOR}")
    return all_results, video_results

def eval_fvd_vae(json_dir, device, submodules_list, **kwargs):
    with open(json_dir, "r") as f:
        video_list = json.load(f)
    video_list = distribute_list_to_rank(video_list)

    model = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    logger.info(f"Model loaded from THUDM/CogVideoX")
    model.enable_slicing()
    model.enable_tiling()
    
    all_results, video_results = compute_fvd_vae_metric(model, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
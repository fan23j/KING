# KING: Kinetic INtegrity and Generation Evaluation

Evaluating the smooth tracking and consistent generation of entities in motion within diffused videos.

## Installation

```bash
pip install -v -e .
```

## Custom Metrics
### 1. Subject Consistency

**Purpose:**  
Subject Consistency measures the ability of a tracking system to maintain consistent identities of tracked entities across frames in a video. High consistency indicates that the system reliably follows each entity without frequent ID switches or tracking losses.

**How It Works:**  
The metric analyzes tracking results by examining the continuity of each tracked subject throughout the video. It calculates the proportion of frames where each subject is consistently tracked without interruption. The overall Subject Consistency score is the average consistency across all tracked subjects in the video.

**Computation Steps:**
1. **Tracking Data Parsing:**  
   Extract tracking information from the results, including frame IDs and track IDs for each entity.

2. **Per-Track Analysis:**  
   For each unique track ID, identify the sequence of frames where the entity appears. Calculate the longest continuous sequence of frames where the track ID is maintained without gaps.

3. **Consistency Score Calculation:**  
   For each track, divide the length of its longest continuous sequence by the total number of frames in the video. The Subject Consistency score for the video is the average of these ratios across all tracks.

**Interpretation:**  
- **Score Range:** 0.0 to 1.0  
- **Higher Scores:** Indicate better tracking consistency, with fewer identity switches and more reliable tracking.
- **Lower Scores:** Suggest frequent tracking interruptions or ID switches, reflecting poorer tracking performance.

### 2. Mean Intersection Over Union (mIOU)

**Purpose:**  
mIOU evaluates the spatial accuracy of the tracking system by comparing the overlap between predicted bounding boxes and ground truth bounding boxes. It quantifies how well the predicted positions of entities align with their actual positions.

**How It Works:**  
For each frame in the video, the IoU between each predicted bounding box and the corresponding ground truth box is calculated. The mean IoU is then averaged across all frames and entities to provide an overall performance metric.

**Computation Steps:**
1. **Ground Truth Loading:**  
   Load ground truth bounding boxes for each video from predefined paths.

2. **Predicted Boxes Processing:**  
   Extract predicted bounding boxes from tracking results and adjust their scales to match the ground truth.

3. **IoU Calculation:**  
   For each frame, compute the IoU between predicted and ground truth boxes using the `box_iou` function from `torchvision`.

4. **Mean IoU Aggregation:**  
   Average the IoU scores across all frames and entities to obtain the mIOU for each video and the overall mIOU across all videos.

**Interpretation:**  
- **Score Range:** 0.0 to 1.0  
- **Higher Scores:** Indicate better spatial alignment between predictions and ground truth.
- **Lower Scores:** Reflect poorer spatial accuracy in tracking.

### 3. Imaging Quality

**Purpose:**  
Imaging Quality assesses the visual fidelity of generated frames in videos. It measures aspects such as sharpness, color accuracy, and overall image quality to ensure that the generated content meets high visual standards.

**How It Works:**  
The metric utilizes the MUSIQ (Multi-scale Image Quality) model to evaluate the quality of each frame in the video. Scores are aggregated to provide an overall Imaging Quality score for the video.

**Computation Steps:**
1. **Video Loading and Preprocessing:**  
   Load video frames and apply necessary transformations, such as resizing and normalization, based on the preprocessing mode.

2. **Quality Scoring:**  
   Pass each preprocessed frame through the MUSIQ model to obtain quality scores.

3. **Aggregation:**  
   Calculate the average score across all frames to derive the Imaging Quality score for the video.

**Interpretation:**  
- **Score Range:** Typically normalized between 0.0 and 1.0  
- **Higher Scores:** Indicate better image quality with clearer, more accurate visuals.
- **Lower Scores:** Suggest lower image quality, possibly due to blurriness, color distortions, or other visual artifacts.

### 4. Frechet Video Distance (FVD) - VAE

**Purpose:**  
FVD measures the distance between the distribution of features from generated videos and real videos. Specifically, the VAE (Variational Autoencoder) variant of FVD evaluates the generative quality by comparing latent representations of videos.

**How It Works:**  
FVD computes the Fréchet Distance between the feature distributions of generated and real videos. Lower FVD scores indicate that the generated videos closely resemble real videos in terms of their feature distributions.

**Computation Steps:**
1. **Feature Extraction:**  
   Encode both generated and ground truth videos using the VAE model to obtain feature representations.

2. **Flattening Features:**  
   Reshape the feature tensors to facilitate covariance computation.

3. **Statistics Calculation:**  
   Compute the mean (`mu`) and covariance (`sigma`) of the features for both generated and real videos.

4. **Fréchet Distance Calculation:**  
   Calculate the Fréchet Distance between the two Gaussian distributions defined by their respective `mu` and `sigma`.

5. **Aggregation:**  
   Average the FVD scores across all videos to obtain the overall FVD-VAE score.

**Interpretation:**  
- **Lower Scores:** Indicate that the generated videos are more similar to real videos in the feature space, reflecting higher generative quality.
- **Higher Scores:** Suggest greater differences between generated and real videos, indicating lower generative fidelity.

### 5. Frechet Video Distance (FVD) - Classifier

**Purpose:**  
Similar to FVD-VAE, FVD-Classifier evaluates the generative quality of videos by comparing feature distributions. However, it utilizes a classifier-based feature extractor (e.g., Timesformer) to capture higher-level semantic features.

**How It Works:**  
FVD-Classifier computes the Fréchet Distance between the distributions of features extracted by a classifier model from generated and real videos. This approach emphasizes semantic consistency and temporal dynamics in the evaluation.

**Computation Steps:**
1. **Video Decoding and Sampling:**  
   Decode videos and sample frames at specified intervals to ensure consistent input for the classifier.

2. **Feature Extraction:**  
   Use a pre-trained Timesformer model to extract semantic features from the sampled frames.

3. **Flattening Features:**  
   Reshape the feature tensors appropriately for covariance computation.

4. **Statistics Calculation:**  
   Compute the mean (`mu`) and covariance (`sigma`) of the features for both generated and real videos.

5. **Fréchet Distance Calculation:**  
   Calculate the Fréchet Distance between the two Gaussian distributions defined by their respective `mu` and `sigma`.

6. **Aggregation:**  
   Average the FVD scores across all videos to obtain the overall FVD-Classifier score.

**Interpretation:**  
- **Lower Scores:** Indicate better semantic and temporal alignment between generated and real videos.
- **Higher Scores:** Reflect greater discrepancies in high-level features, suggesting lower generative quality.

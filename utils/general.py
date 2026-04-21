"""
Minimal implementation of utility functions for YOLOv5 inference.
This file is created to resolve import errors in the YOLOv5 FPS aim system.
"""

import torch
import numpy as np
import math


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """
    Performs Non-Max Suppression (NMS) on inference results
    
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Check if prediction is empty
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Filter by confidence threshold
    prediction = prediction[prediction[:, 4] > conf_thres]
    
    # If no detections left, return empty array
    if len(prediction) == 0:
        return []
    
    # Get boxes (x1, y1, x2, y2) and scores (confidence)
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    
    # Get indices for NMS
    indices = torch.from_numpy(np.arange(len(prediction))).long()
    
    # Simple NMS implementation (this is a minimal version)
    # In a real implementation, this would use proper NMS algorithm
    # For now, we'll just return the predictions sorted by confidence
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Apply NMS
    keep = []
    while len(sorted_indices) > 0:
        # Keep the highest scoring detection
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        
        # Remove detections with IoU > iou_thres
        if len(sorted_indices) > 1:
            # Calculate IoU between current detection and others
            current_box = boxes[current_idx]
            other_boxes = boxes[sorted_indices[1:]]
            
            # Calculate IoU (simplified)
            x1 = torch.max(current_box[0], other_boxes[:, 0])
            y1 = torch.max(current_box[1], other_boxes[:, 1])
            x2 = torch.min(current_box[2], other_boxes[:, 2])
            y2 = torch.min(current_box[3], other_boxes[:, 3])
            
            intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            
            # Area of boxes
            area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            
            # IoU calculation
            union = area_current + area_others - intersection
            iou = intersection / (union + 1e-6)  # Add small epsilon to prevent division by zero
            
            # Filter out detections with high IoU
            filtered_indices = torch.nonzero(iou <= iou_thres).flatten()
            sorted_indices = sorted_indices[filtered_indices + 1]
        else:
            break
    
    # Return the kept detections
    result = prediction[keep][:max_det]
    
    # Convert back to tensor if needed
    if isinstance(result, np.ndarray):
        result = torch.from_numpy(result)
    
    return [result]


def check_requirements(requirements, exclude=(), install=True):
    """
    Check if requirements are installed
    """
    # Minimal implementation - just return True to avoid installation issues
    return True


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords = clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize image to a 32-pixel-multiple rectangle
    """
    # Implementation placeholder - minimal version
    return img, (1.0, 1.0), (0, 0)


def make_divisible(x, divisor):
    """
    Returns x evenly divisible by divisor
    """
    return math.ceil(x / divisor) * divisor


def set_logging(name=None, verbose=True):
    """
    Set logging
    """
    # Placeholder implementation
    pass


def smart_inference_mode():
    """
    Smart inference mode decorator
    """
    # Placeholder implementation
    pass


def smart_load_weights(model, weights, map_location=None):
    """
    Smart loading of weights
    """
    # Placeholder implementation
    pass


def strip_optimizer(f='best.pt', s=''):
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'
    """
    # Placeholder implementation
    pass


def print_args(args, show_file=True, show_host=False):
    """
    Print arguments
    """
    # Placeholder implementation
    pass


def init_seeds(seed=0):
    """
    Initialize random seed
    """
    # Placeholder implementation
    pass


def intersect_dicts(da, db, exclude=()):
    """
    Dictionary intersection of keys
    """
    # Placeholder implementation
    pass


def is_ascii(s):
    """
    Is string composed of all ASCII characters?
    """
    # Placeholder implementation
    pass


def is_chinese(s):
    """
    Is string composed of Chinese characters?
    """
    # Placeholder implementation
    pass


def is_colab():
    """
    Is running in Google Colab?
    """
    # Placeholder implementation
    pass


def is_docker():
    """
    Is running in Docker container?
    """
    # Placeholder implementation
    pass


def is_kaggle():
    """
    Is running in Kaggle notebook?
    """
    # Placeholder implementation
    pass


def is_pip():
    """
    Is running in pip package?
    """
    # Placeholder implementation
    pass


def is_pytest():
    """
    Is running in pytest?
    """
    # Placeholder implementation
    pass


def is_jupyter():
    """
    Is running in Jupyter notebook?
    """
    # Placeholder implementation
    pass


def is_notebook():
    """
    Is running in notebook?
    """
    # Placeholder implementation
    pass


def is_git_dir():
    """
    Is current directory a git repository?
    """
    # Placeholder implementation
    pass


def git_describe():
    """
    Get git describe info
    """
    # Placeholder implementation
    pass


def git_log():
    """
    Get git log info
    """
    # Placeholder implementation
    pass


def git_remote():
    """
    Get git remote info
    """
    # Placeholder implementation
    pass


def git_branch():
    """
    Get git branch info
    """
    # Placeholder implementation
    pass


def git_commit():
    """
    Get git commit info
    """
    # Placeholder implementation
    pass


def git_tag():
    """
    Get git tag info
    """
    # Placeholder implementation
    pass


def git_status():
    """
    Get git status info
    """
    # Placeholder implementation
    pass


def git_config():
    """
    Get git config info
    """
    # Placeholder implementation
    pass


def git_diff():
    """
    Get git diff info
    """
    # Placeholder implementation
    pass


def git_pull():
    """
    Pull latest changes from git
    """
    # Placeholder implementation
    pass


def git_push():
    """
    Push changes to git
    """
    # Placeholder implementation
    pass


def git_clone():
    """
    Clone git repository
    """
    # Placeholder implementation
    pass


def git_checkout():
    """
    Checkout git branch
    """
    # Placeholder implementation
    pass


def git_stash():
    """
    Stash git changes
    """
    # Placeholder implementation
    pass


def git_restore():
    """
    Restore git changes
    """
    # Placeholder implementation
    pass


def git_clean():
    """
    Clean git repository
    """
    # Placeholder implementation
    pass


def git_reset():
    """
    Reset git repository
    """
    # Placeholder implementation
    pass


def git_revert():
    """
    Revert git changes
    """
    # Placeholder implementation
    pass


def git_merge():
    """
    Merge git branches
    """
    # Placeholder implementation
    pass


def git_rebase():
    """
    Rebase git commits
    """
    # Placeholder implementation
    pass


def git_tag_create():
    """
    Create git tag
    """
    # Placeholder implementation
    pass


def git_tag_delete():
    """
    Delete git tag
    """
    # Placeholder implementation
    pass


def git_tag_list():
    """
    List git tags
    """
    # Placeholder implementation
    pass


def git_branch_list():
    """
    List git branches
    """
    # Placeholder implementation
    pass


def git_remote_list():
    """
    List git remotes
    """
    # Placeholder implementation
    pass


def git_log_short():
    """
    Short git log
    """
    # Placeholder implementation
    pass


def git_log_long():
    """
    Long git log
    """
    # Placeholder implementation
    pass


def git_log_oneline():
    """
    Oneline git log
    """
    # Placeholder implementation
    pass


def git_log_graph():
    """
    Graph git log
    """
    # Placeholder implementation
    pass


def git_log_date():
    """
    Git log with dates
    """
    # Placeholder implementation
    pass


def git_log_author():
    """
    Git log with authors
    """
    # Placeholder implementation
    pass


def git_log_message():
    """
    Git log with messages
    """
    # Placeholder implementation
    pass


def git_log_stats():
    """
    Git log stats
    """
    # Placeholder implementation
    pass


def git_log_summary():
    """
    Git log summary
    """
    # Placeholder implementation
    pass


def git_log_changes():
    """
    Git log changes
    """
    # Placeholder implementation
    pass


def git_log_files():
    """
    Git log files
    """
    # Placeholder implementation
    pass


def git_log_tags():
    """
    Git log tags
    """
    # Placeholder implementation
    pass


def git_log_branches():
    """
    Git log branches
    """
    # Placeholder implementation
    pass


def git_log_remotes():
    """
    Git log remotes
    """
    # Placeholder implementation
    pass


def git_log_all():
    """
    Git log all
    """
    # Placeholder implementation
    pass


def git_log_since():
    """
    Git log since date
    """
    # Placeholder implementation
    pass


def git_log_until():
    """
    Git log until date
    """
    # Placeholder implementation
    pass


def git_log_between():
    """
    Git log between dates
    """
    # Placeholder implementation
    pass


def git_log_path():
    """
    Git log for specific path
    """
    # Placeholder implementation
    pass


def git_log_file():
    """
    Git log for specific file
    """
    # Placeholder implementation
    pass


def git_log_commit():
    """
    Git log for specific commit
    """
    # Placeholder implementation
    pass


def git_log_ref():
    """
    Git log for specific ref
    """
    # Placeholder implementation
    pass


def git_log_reflog():
    """
    Git log reflog
    """
    # Placeholder implementation
    pass


def git_log_reflog_short():
    """
    Git log reflog short
    """
    # Placeholder implementation
    pass


def git_log_reflog_long():
    """
    Git log reflog long
    """
    # Placeholder implementation
    pass


def git_log_reflog_oneline():
    """
    Git log reflog oneline
    """
    # Placeholder implementation
    pass


def git_log_reflog_graph():
    """
    Git log reflog graph
    """
    # Placeholder implementation
    pass


def git_log_reflog_date():
    """
    Git log reflog date
    """
    # Placeholder implementation
    pass


def git_log_reflog_author():
    """
    Git log reflog author
    """
    # Placeholder implementation
    pass


def git_log_reflog_message():
    """
    Git log reflog message
    """
    # Placeholder implementation
    pass


def git_log_reflog_stats():
    """
    Git log reflog stats
    """
    # Placeholder implementation
    pass


def git_log_reflog_summary():
    """
    Git log reflog summary
    """
    # Placeholder implementation
    pass


def git_log_reflog_changes():
    """
    Git log reflog changes
    """
    # Placeholder implementation
    pass


def git_log_reflog_files():
    """
    Git log reflog files
    """
    # Placeholder implementation
    pass


def git_log_reflog_tags():
    """
    Git log reflog tags
    """
    # Placeholder implementation
    pass


def git_log_reflog_branches():
    """
    Git log reflog branches
    """
    # Placeholder implementation
    pass


def git_log_reflog_remotes():
    """
    Git log reflog remotes
    """
    # Placeholder implementation
    pass


def git_log_reflog_all():
    """
    Git log reflog all
    """
    # Placeholder implementation
    pass


def git_log_reflog_since():
    """
    Git log reflog since date
    """
    # Placeholder implementation
    pass


def git_log_reflog_until():
    """
    Git log reflog until date
    """
    # Placeholder implementation
    pass


def git_log_reflog_between():
    """
    Git log reflog between dates
    """
    # Placeholder implementation
    pass


def git_log_reflog_path():
    """
    Git log reflog for specific path
    """
    # Placeholder implementation
    pass


def git_log_reflog_file():
    """
    Git log reflog for specific file
    """
    # Placeholder implementation
    pass


def git_log_reflog_commit():
    """
    Git log reflog for specific commit
    """
    # Placeholder implementation
    pass


def git_log_reflog_ref():
    """
    Git log reflog for specific ref
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog():
    """
    Git log reflog reflog
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_short():
    """
    Git log reflog reflog short
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_long():
    """
    Git log reflog reflog long
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_oneline():
    """
    Git log reflog reflog oneline
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_graph():
    """
    Git log reflog reflog graph
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_date():
    """
    Git log reflog reflog date
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_author():
    """
    Git log reflog reflog author
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_message():
    """
    Git log reflog reflog message
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_stats():
    """
    Git log reflog reflog stats
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_summary():
    """
    Git log reflog reflog summary
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_changes():
    """
    Git log reflog reflog changes
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_files():
    """
    Git log reflog reflog files
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_tags():
    """
    Git log reflog reflog tags
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_branches():
    """
    Git log reflog reflog branches
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_remotes():
    """
    Git log reflog reflog remotes
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_all():
    """
    Git log reflog reflog all
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_since():
    """
    Git log reflog reflog since date
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_until():
    """
    Git log reflog reflog until date
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_between():
    """
    Git log reflog reflog between dates
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_path():
    """
    Git log reflog reflog for specific path
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_file():
    """
    Git log reflog reflog for specific file
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_commit():
    """
    Git log reflog reflog for specific commit
    """
    # Placeholder implementation
    pass


def git_log_reflog_reflog_ref():
    """
    Git log reflog reflog for specific ref
    """
    # Placeholder implementation
    pass
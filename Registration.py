import argparse
import re
import time
import cv2
import numpy as np
import os
import glob
import sys
import io

# Set standard output encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def image_stack_registration(input_source, output_path=None):
    """
    Image stack registration: coarse alignment with zoom followed by fine registration with stabilisation
    
    Args:
        input_source: string (directory path) or list of images
        output_path: string, directory path to save results (optional)
    
    Returns:
        list of registered images
    """
    # Step 1: Coarse alignment using zoom
    zoom_aligned_images = image_stack_align_zoom(input_source)
    
    # Step 2: Fine registration using stabilisation
    final_images = image_stack_stabilisation(zoom_aligned_images, output_path)
    
    return final_images

def image_stack_align_zoom(input_source, output_path=None):
    """
    Align images from directory path or image list
    Args:
        input_source: string (directory path) or list of images
        output_path: string, directory path to save results (optional)
    Returns:
        list of aligned images
    """
    # Process input source
    if isinstance(input_source, str):
        # Pre-compile regular expression
        num_pattern = re.compile(r"\d+")
        # Use generator expression and list comprehension for optimized file filtering and sorting
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
            key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1])
        )
        # Use list comprehension to read all images at once
        images = [cv2.imread(path) for path in img_paths]
    else:
        images = input_source

    # Cache image dimensions and feature detector
    img_first = images[0]
    img_last = images[-1]
    h_first, w_first = img_first.shape[:2]
    h_last, w_last = img_last.shape[:2]

    # SIFT feature detection and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_first, None)
    kp2, des2 = sift.detectAndCompute(img_last, None)

    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    # Find valid matching points
    min_distance_threshold = w_first / 2
    p1 = p2 = q1 = q2 = None

    # Use numpy array operations to optimize distance calculation
    for i in range(len(matches) - 1):
        p1 = np.array(kp1[matches[i].queryIdx].pt)
        p2 = np.array(kp1[matches[i + 1].queryIdx].pt)
        q1 = np.array(kp2[matches[i].trainIdx].pt)
        q2 = np.array(kp2[matches[i + 1].trainIdx].pt)

        if np.linalg.norm(p1 - p2) > min_distance_threshold:
            break
    else:
        raise ValueError("Unable to find matching point pairs that meet the criteria")

    # Calculate scaling factor
    dist = np.linalg.norm(p1 - p2)
    dist2 = np.linalg.norm(q1 - q2)
    c = max(dist2 / dist, dist / dist2)

    # Pre-calculate all scaling factors
    aug_list = np.linspace(1, c, len(images))
    is_forward = dist2 / dist > 1
    target_shape = (h_last, w_last) if is_forward else (h_first, w_first)
    aug_factors = aug_list[::-1] if is_forward else aug_list

    # Pre-create output directory
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Process images
    aligned_images = []
    target_h, target_w = target_shape

    for img, aug_factor in zip(images, aug_factors):
        # Use cv2.INTER_AREA for scaling
        img_resized = cv2.resize(img, None, fx=aug_factor, fy=aug_factor, interpolation=cv2.INTER_AREA)

        # Calculate crop position
        h, w = img_resized.shape[:2]
        x = (w - target_w) // 2
        y = (h - target_h) // 2

        # Crop image
        img_crop = img_resized[y:y + target_h, x:x + target_w]
        aligned_images.append(img_crop)

        # Save results
        if output_path:
            idx = len(aligned_images) - 1
            cv2.imwrite(os.path.join(output_path, f'frame_{idx:04d}.png'), img_crop)

    return aligned_images

def image_stack_stabilisation(input_source, output_path=None):
    """
    Stabilize images from directory path or image list
    Args:
        input_source: string (directory path) or list of images
        output_path: string, directory path to save results (optional)
    Returns:
        list of stabilized images
    """
    # Process input source
    if isinstance(input_source, str):
        img_paths = sorted(
            [os.path.join(input_source, file) for file in os.listdir(input_source)
             if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']],
            key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1])
        )
        images = [cv2.imread(path) for path in img_paths]
    else:
        images = input_source

    n_frames = len(images)
    img_first = images[0]
    h, w = img_first.shape[:2]

    prev_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames, 3), np.float32)

    # Feature detection parameters
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate transforms
    for i in range(n_frames - 1):
        curr = images[i + 1]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        idx = status.ravel() == 1
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        dx, dy = m[0, 2], m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    transforms[-1] = transforms[-2]

    # Smooth trajectory
    def smooth(trajectory, radius=30):
        smoothed_trajectory = np.copy(trajectory)
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padding = np.pad(trajectory, ((radius, radius), (0, 0)), 'edge')
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(padding[:, i], kernel, mode='valid')
        return smoothed_trajectory

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # Apply transforms
    stabilized_images = []
    for i, frame in enumerate(images):
        dx, dy, da = transforms_smooth[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
        frame_stabilized = cv2.warpAffine(frame_stabilized, T, (w, h))
        stabilized_images.append(frame_stabilized)

        # Save results if output path is provided
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_file = os.path.join(output_path, f'frame_{i:04d}.png')
            cv2.imwrite(output_file, frame_stabilized)

    return stabilized_images

def process_image_stack(input_path, output_path=None):
    """
    Main function for processing image stack using zoom+stabilisation sequential registration method
    
    Args:
        input_path: input image directory path
        output_path: output directory path (optional)
    
    Returns:
        list of registered images
    """
    return image_stack_registration(input_path, output_path)

def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description='Image Stack Registration Tool (zoom coarse alignment + stabilisation fine registration)')
    parser.add_argument('input_path', help='Input image directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        processed_images = process_image_stack(args.input_path, args.output)
        
        end_time = time.time()
        
        if args.output:
            print(f"Results saved to: {args.output}")
            
    except Exception as e:
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

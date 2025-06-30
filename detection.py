import cv2
import numpy as np
import argparse
import os
import glob # For wildcard expansion
import math # Not explicitly used, but good for general math ops

def non_max_suppression_fast(boxes, overlapThresh):
    """
    A fast approximate Non-Maximum Suppression algorithm.
    This function removes redundant overlapping bounding boxes.
    Adapted from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        boxes (np.array): A list of bounding boxes in the format [x1, y1, x2, y2]
        overlapThresh (float): The maximum allowed overlap ratio (Intersection Over Union).

    Returns:
        list: The indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2) # Sort by bottom-right y-coordinate

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return pick

def detect_with_template_matching(template_path, target_path, threshold, scales, match_method_name):
    """
    Performs multi-scale template matching.
    Returns a list of [x1, y1, x2, y2, score] detections.
    """
    template_orig = cv2.imread(template_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)

    if template_orig is None or target is None:
        return [], template_orig, target # Return empty detections and images for error handling

    target_img_height, target_img_width = target.shape[:2]

    # Map string method name to OpenCV constant
    match_methods = {
        'TM_SQDIFF': cv2.TM_SQDIFF,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
        'TM_CCOEFF': cv2.TM_CCOEFF,
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    }
    match_method = match_methods.get(match_method_name, cv2.TM_CCOEFF_NORMED) # Default if invalid

    all_detections = [] # Stores [x1, y1, x2, y2, score]

    for scale in scales:
        resized_template_w = int(template_orig.shape[1] * scale)
        resized_template_h = int(template_orig.shape[0] * scale)

        if resized_template_w < 10 or resized_template_h < 10:
            continue
        if resized_template_w > target_img_width or resized_template_h > target_img_height:
            continue

        resized_template = cv2.resize(template_orig, (resized_template_w, resized_template_h))

        res = cv2.matchTemplate(target, resized_template, match_method)

        if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc_y, loc_x = np.where(res <= threshold)
        else:
            loc_y, loc_x = np.where(res >= threshold)

        for (x, y) in zip(loc_x, loc_y):
            score = res[y, x]
            all_detections.append([x, y, x + resized_template_w, y + resized_template_h, score])

    return all_detections, template_orig, target

def detect_with_feature_matching(template_path, target_path, min_match_count=10):
    """
    Performs feature-based matching using ORB and Homography.
    Returns a list of [x1, y1, x2, y2, score] detections (currently one per target image).
    """
    template_orig = cv2.imread(template_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)

    if template_orig is None or target is None:
        return [], template_orig, target

    template_gray = cv2.cvtColor(template_orig, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8) # Increased features for more robustness

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(target_gray, None)

    if des1 is None or des2 is None:
        print("  Warning: No descriptors found for template or target image using ORB. Skipping feature matching.")
        return [], template_orig, target

    # Create BFMatcher object
    # NORM_HAMMING for ORB's binary descriptors
    # crossCheck=True ensures best matches in both directions (A->B and B->A), improves accuracy
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance. Less distance = better match.
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > min_match_count:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography - M is the transformation matrix
        # RANSAC is used to filter outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = template_gray.shape
            # Define corners of the template image
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # Transform corners to the target image perspective
            dst_corners = cv2.perspectiveTransform(pts, M)

            # Convert transformed corners to a bounding box [x1, y1, x2, y2]
            # This is a potentially rotated rectangle, so we find its min/max
            x_coords = dst_corners[:, 0, 0]
            y_coords = dst_corners[:, 0, 1]

            x1 = int(np.min(x_coords))
            y1 = int(np.min(y_coords))
            x2 = int(np.max(x_coords))
            y2 = int(np.max(y_coords))

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(target.shape[1] - 1, x2)
            y2 = min(target.shape[0] - 1, y2)

            # A simple "score" for feature matching can be the number of inliers or just a high value.
            # We'll use the number of good matches for a relative "score" here.
            score = len(matches) / max(len(kp1), len(kp2)) # Normalize by total keypoints for a relative score
            
            return [[x1, y1, x2, y2, score]], template_orig, target
        else:
            print("  Warning: Homography could not be found. Not enough reliable matches.")
    else:
        print(f"  Warning: Not enough matches ({len(matches)} < {min_match_count}) found for feature matching. Skipping.")
    
    return [], template_orig, target


def main():
    parser = argparse.ArgumentParser(
        description="Detects an object from a template image in various target images. "
                    "Supports both multi-scale template matching and feature-based matching (ORB)."
                    "Generates annotated images and YOLO-style annotation text files."
    )
    parser.add_argument(
        'template_image',
        type=str,
        help='Path to the template image (the object you want to detect).'
    )
    parser.add_argument(
        'target_images_pattern',
        type=str,
        help='Path pattern to one or more target images (e.g., "images/*.jpg" or "image1.jpg image2.png"). '
             'Enclose patterns with wildcards in quotes.'
    )
    parser.add_argument(
        '--detection_method',
        type=str,
        default='template_matching',
        choices=['template_matching', 'feature_matching'],
        help='Method to use for object detection: "template_matching" (default) or "feature_matching" (more robust to variations).'
    )
    parser.add_argument(
        '--output_img_dir',
        type=str,
        default='detected_images',
        help='Directory to save the annotated images (default: detected_images/).'
    )
    parser.add_argument(
        '--annotations_dir',
        type=str,
        default='yolo_annotations',
        help='Directory to save the YOLO-style annotation text files (default: yolo_annotations/).'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=0,
        help='Class ID to use in YOLO annotation files (default: 0).'
    )
    parser.add_argument(
        '--bbox_color',
        type=lambda s: tuple(map(int, s.split(','))),
        default='0,0,255', # Red in BGR
        help='BGR color for bounding boxes (e.g., "0,0,255" for red, "255,0,0" for blue). Separate values with commas.'
    )
    parser.add_argument(
        '--bbox_thickness',
        type=int,
        default=2,
        help='Thickness of bounding box lines (default: 2).'
    )

    # Arguments specific to template matching
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for template matching (0.0 to 1.0). For TM_SQDIFF/TM_SQDIFF_NORMED, lower is better. '
             'For others, higher is better. Adjust if too many or too few detections (default: 0.7).'
    )
    parser.add_argument(
        '--min_scale',
        type=float,
        default=0.1,
        help='Minimum scale factor for the template (e.g., 0.1 for 10%% of original size). Used with template_matching.'
    )
    parser.add_argument(
        '--max_scale',
        type=float,
        default=1.0,
        help='Maximum scale factor for the template (e.g., 1.0 for 100%% of original size). Used with template_matching.'
    )
    parser.add_argument(
        '--num_scales',
        type=int,
        default=20,
        help='Number of scales to sample between min_scale and max_scale. Used if --scale_step is not set (default: 20). Used with template_matching.'
    )
    parser.add_argument(
        '--scale_step',
        type=float,
        help='Step size between scales (e.g., 0.05 for 5%% increments). If set, overrides --num_scales. Used with template_matching.'
    )
    parser.add_argument(
        '--match_method',
        type=str,
        default='TM_CCOEFF_NORMED',
        choices=['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED'],
        help='OpenCV template matching method to use. TM_CCOEFF_NORMED is generally recommended (default: TM_CCOEFF_NORMED).'
             ' Only used with "template_matching" method.'
    )

    # Arguments specific to feature matching
    parser.add_argument(
        '--min_feature_matches',
        type=int,
        default=10,
        help='Minimum number of good feature matches required to consider a detection valid. Only used with "feature_matching" method (default: 10).'
    )

    # General output control arguments
    parser.add_argument(
        '--overlap_threshold',
        type=float,
        default=0.3, # IOU threshold for NMS
        help='Intersection Over Union (IOU) threshold for Non-Maximum Suppression. Higher values keep more overlapping boxes (default: 0.3).'
    )
    parser.add_argument(
        '--no_save_annotated_images',
        action='store_true',
        help='Do not save the annotated image files.'
    )
    parser.add_argument(
        '--no_save_annotations',
        action='store_true',
        help='Do not save the YOLO annotation text files.'
    )
    parser.add_argument(
        '--output_image_suffix',
        type=str,
        default='_detected',
        help='Suffix to add to the base filename of the annotated images (e.g., "_detected").'
             ' If empty, original file name is used (overwrites original if in same directory). (default: "_detected")'
    )


    args = parser.parse_args()

    # Resolve target image paths using glob
    target_image_paths = []
    for part in args.target_images_pattern.split():
        target_image_paths.extend(glob.glob(part))

    if not target_image_paths:
        print(f"Error: No target images found matching pattern '{args.target_images_pattern}'. Please check the path/pattern.")
        return

    # Create output directories if they don't exist, only if saving is enabled
    if args.no_save_annotated_images and args.no_save_annotations:
        print("Warning: Neither annotated images nor annotation files will be saved (both --no_save_annotated_images and --no_save_annotations are set).")
    else:
        if args.no_save_annotated_images:
            print(f"Output directory for images will not be created (as --no_save_annotated_images is set).")
        else:
            os.makedirs(args.output_img_dir, exist_ok=True)
            print(f"Created output directory for images: '{args.output_img_dir}'")

        if args.no_save_annotations:
            print(f"Output directory for annotations will not be created (as --no_save_annotations is set).")
        else:
            os.makedirs(args.annotations_dir, exist_ok=True)
            print(f"Created output directory for annotations: '{args.annotations_dir}'")


    print("\n--- Starting Flexible Object Detection ---")
    print(f"Detection Method: '{args.detection_method}'")
    print(f"Template Image: '{args.template_image}'")
    print(f"Target Images Pattern: '{args.target_images_pattern}'")
    print(f"Number of Target Images to Process: {len(target_image_paths)}")
    print(f"Output Image Directory: '{args.output_img_dir}' (Saving: {not args.no_save_annotated_images})")
    print(f"Output Annotation Directory: '{args.annotations_dir}' (Saving: {not args.no_save_annotations})")
    print(f"YOLO Class ID: {args.class_id}")
    print(f"Bounding Box Color (BGR): {args.bbox_color}, Thickness: {args.bbox_thickness}")
    print(f"NMS Overlap Threshold (IOU): {args.overlap_threshold}")
    print(f"Output Image Suffix: '{args.output_image_suffix}'")

    if args.detection_method == 'template_matching':
        # Generate scales list for template matching
        if args.scale_step:
            scales_list = np.arange(args.min_scale, args.max_scale + args.scale_step / 2, args.scale_step)[::-1]
        else:
            scales_list = np.linspace(args.min_scale, args.max_scale, args.num_scales)[::-1]
        print(f"Template Matching Specifics:")
        print(f"  Detection Threshold: {args.threshold}")
        print(f"  Scale Range: {args.min_scale:.2f} to {args.max_scale:.2f}")
        if args.scale_step:
            print(f"  Scales Step: {args.scale_step:.2f} ({len(scales_list)} steps)")
        else:
            print(f"  Number of Scales: {args.num_scales} ({len(scales_list)} steps)")
        print(f"  Template Matching Method: {args.match_method}")
    elif args.detection_method == 'feature_matching':
        print(f"Feature Matching Specifics:")
        print(f"  Minimum Feature Matches: {args.min_feature_matches}")
        # Note: threshold for feature matching is implicitly handled by RANSAC and min_match_count
        scales_list = [] # Not used in feature matching, but passed to keep function signature consistent
    
    print("----------------------------------------------------\n")

    for i, target_image_path in enumerate(target_image_paths):
        print(f"Processing target image {i+1}/{len(target_image_paths)}: '{os.path.basename(target_image_path)}'")
        
        all_detections_list = [] # Renamed to clearly indicate it's a list from detection functions
        template_img_ref = None
        target_img_ref = None

        if args.detection_method == 'template_matching':
            all_detections_list, template_img_ref, target_img_ref = detect_with_template_matching(
                args.template_image, target_image_path, args.threshold, scales_list, args.match_method
            )
        elif args.detection_method == 'feature_matching':
            all_detections_list, template_img_ref, target_img_ref = detect_with_feature_matching(
                args.template_image, target_image_path, args.min_feature_matches
            )
        
        if template_img_ref is None or target_img_ref is None:
            print(f"  Skipping '{os.path.basename(target_image_path)}' due to image loading errors.")
            print("-" * 30)
            continue

        yolo_annotations = []
        num_detections = 0

        # --- FIX START ---
        # Convert the list of detections to a NumPy array *before* NMS and drawing.
        all_detections_np = np.array(all_detections_list) 

        if len(all_detections_np) > 0: # Check if there are any detections after conversion
            # Apply NMS to the detections from either method
            # boxes_for_nms is a view of all_detections_np's first 4 columns
            boxes_for_nms = all_detections_np[:, :4] 
            pick = non_max_suppression_fast(boxes_for_nms, args.overlap_threshold)

            for idx in pick:
                # Now access all_detections_np (the NumPy array)
                x1, y1, x2, y2 = all_detections_np[idx][:4].astype(int)
                num_detections += 1

                # Calculate YOLO format coordinates
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                norm_center_x = center_x / target_img_ref.shape[1]
                norm_center_y = center_y / target_img_ref.shape[0]
                norm_width = box_width / target_img_ref.shape[1]
                norm_height = box_height / target_img_ref.shape[0]

                yolo_annotations.append(f"{args.class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

                if not args.no_save_annotated_images:
                    # Draw rectangle on the *original* target image reference, not a copy
                    cv2.rectangle(target_img_ref, (x1, y1), (x2, y2), args.bbox_color, args.bbox_thickness)
        # --- FIX END ---

        print(f"  Detected {num_detections} instances of the object in {os.path.basename(target_image_path)}")

        # Save the annotated image
        if not args.no_save_annotated_images:
            output_filename = f"{os.path.splitext(os.path.basename(target_image_path))[0]}{args.output_image_suffix}{os.path.splitext(os.path.basename(target_image_path))[1]}"
            output_path = os.path.join(args.output_img_dir, output_filename)
            cv2.imwrite(output_path, target_img_ref)
            print(f"  Annotated image saved to '{output_path}'")
        else:
            print("  Skipping saving annotated image (as requested).")

        # Save the YOLO annotations
        if not args.no_save_annotations:
            annotation_filename = f"{os.path.splitext(os.path.basename(target_image_path))[0]}.txt"
            annotation_path = os.path.join(args.annotations_dir, annotation_filename)
            with open(annotation_path, 'w') as f:
                for annotation_line in yolo_annotations:
                    f.write(annotation_line + '\n')
            print(f"  YOLO annotations saved to '{annotation_path}'")
        else:
            print("  Skipping saving YOLO annotations (as requested).")

        print("-" * 30) # Separator

    print("\n--- Detection Complete ---")
    if not args.no_save_annotated_images:
        print(f"Annotated images saved to the '{args.output_img_dir}' directory.")
    if not args.no_save_annotations:
        print(f"YOLO annotations saved to the '{args.annotations_dir}' directory.")

if __name__ == "__main__":
    main()
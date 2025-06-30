import cv2
import numpy as np
import argparse
import os
import glob # For listing files and wildcard expansion

def analyze_yolo_annotations(annotations_dir, reference_image_width=None, reference_image_height=None):
    """
    Analyzes YOLO annotation files to find average/median bounding box parameters.

    Args:
        annotations_dir (str): Path to the directory containing YOLO .txt annotation files.
        reference_image_width (int, optional): Width of a reference image to convert
                                               normalized coordinates to pixel values for logging.
        reference_image_height (int, optional): Height of a reference image.

    Returns:
        dict: A dictionary containing average and median bounding box parameters,
              and optionally pixel coordinates for a reference image. Returns None if no annotations.
    """
    all_x_centers = []
    all_y_centers = []
    all_widths = []
    all_heights = []
    total_annotations_processed = 0
    files_processed = 0
    files_with_errors = []

    annotation_files = glob.glob(os.path.join(annotations_dir, '*.txt'))

    if not annotation_files:
        print(f"No .txt annotation files found in '{annotations_dir}'. Please check the directory path.")
        return None

    print(f"Scanning {len(annotation_files)} annotation files in '{annotations_dir}' for analysis...")

    for file_path in annotation_files:
        files_processed += 1
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                # print(f"  Warning: Annotation file '{os.path.basename(file_path)}' is empty. Skipping.") # Too chatty
                continue

            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) == 5:
                    try:
                        # We only need the normalized coordinates for calculation
                        norm_x_center = float(parts[1])
                        norm_y_center = float(parts[2])
                        norm_width = float(parts[3])
                        norm_height = float(parts[4])

                        all_x_centers.append(norm_x_center)
                        all_y_centers.append(norm_y_center)
                        all_widths.append(norm_width)
                        all_heights.append(norm_height)
                        total_annotations_processed += 1
                    except ValueError:
                        print(f"  Warning: Invalid numeric value in line '{line.strip()}' in '{os.path.basename(file_path)}'. Skipping line.")
                        if os.path.basename(file_path) not in files_with_errors:
                            files_with_errors.append(os.path.basename(file_path))
                else:
                    print(f"  Warning: Invalid format in line '{line.strip()}' in '{os.path.basename(file_path)}'. Expected 5 values. Skipping line.")
                    if os.path.basename(file_path) not in files_with_errors:
                        files_with_errors.append(os.path.basename(file_path))

        except Exception as e:
            print(f"  Error reading file '{os.path.basename(file_path)}': {e}. Skipping file.")
            files_with_errors.append(os.path.basename(file_path))

    if total_annotations_processed == 0:
        print("\nNo valid annotations were processed. Unable to calculate ideal position.")
        return None

    # Calculate Averages (Mean)
    avg_x_center = np.mean(all_x_centers)
    avg_y_center = np.mean(all_y_centers)
    avg_width = np.mean(all_widths)
    avg_height = np.mean(all_heights)

    # Calculate Medians
    median_x_center = np.median(all_x_centers)
    median_y_center = np.median(all_y_centers)
    median_width = np.median(all_widths)
    median_height = np.median(all_heights)
    
    # Calculate Standard Deviations (optional, for variability)
    std_x_center = np.std(all_x_centers)
    std_y_center = np.std(all_y_centers)
    std_width = np.std(all_widths)
    std_height = np.std(all_heights)

    results = {
        'total_files_scanned': files_processed,
        'files_with_errors': files_with_errors,
        'total_annotations_processed': total_annotations_processed,
        'average_normalized_bbox': {
            'x_center': avg_x_center,
            'y_center': avg_y_center,
            'width': avg_width,
            'height': avg_height
        },
        'median_normalized_bbox': {
            'x_center': median_x_center,
            'y_center': median_y_center,
            'width': median_width,
            'height': median_height
        },
        'std_dev_normalized_bbox': {
            'x_center': std_x_center,
            'y_center': std_y_center,
            'width': std_width,
            'height': std_height
        }
    }

    # Convert to pixel coordinates if reference image dimensions are provided
    if reference_image_width is not None and reference_image_height is not None:
        # For Average
        avg_pixel_x_center = int(avg_x_center * reference_image_width)
        avg_pixel_y_center = int(avg_y_center * reference_image_height)
        avg_pixel_width = int(avg_width * reference_image_width)
        avg_pixel_height = int(avg_height * reference_image_height)

        avg_pixel_x1 = avg_pixel_x_center - avg_pixel_width // 2
        avg_pixel_y1 = avg_pixel_y_center - avg_pixel_height // 2
        avg_pixel_x2 = avg_pixel_x1 + avg_pixel_width
        avg_pixel_y2 = avg_pixel_y1 + avg_pixel_height

        results['average_pixel_bbox_for_reference_image'] = {
            'x_center': avg_pixel_x_center,
            'y_center': avg_pixel_y_center,
            'width': avg_pixel_width,
            'height': avg_pixel_height,
            'x1': avg_pixel_x1,
            'y1': avg_pixel_y1,
            'x2': avg_pixel_x2,
            'y2': avg_pixel_y2,
            'reference_width': reference_image_width,
            'reference_height': reference_image_height
        }

        # For Median
        median_pixel_x_center = int(median_x_center * reference_image_width)
        median_pixel_y_center = int(median_y_center * reference_image_height)
        median_pixel_width = int(median_width * reference_image_width)
        median_pixel_height = int(median_height * reference_image_height)

        median_pixel_x1 = median_pixel_x_center - median_pixel_width // 2
        median_pixel_y1 = median_pixel_y_center - median_pixel_height // 2
        median_pixel_x2 = median_pixel_x1 + median_pixel_width
        median_pixel_y2 = median_pixel_y1 + median_pixel_height

        results['median_pixel_bbox_for_reference_image'] = {
            'x_center': median_pixel_x_center,
            'y_center': median_pixel_y_center,
            'width': median_pixel_width,
            'height': median_pixel_height,
            'x1': median_pixel_x1,
            'y1': median_pixel_y1,
            'x2': median_pixel_x2,
            'y2': median_pixel_y2,
            'reference_width': reference_image_width,
            'reference_height': reference_image_height
        }

    return results

def place_logo_on_image(target_image_path, logo_path, normalized_bbox, output_image_path):
    """
    Places a logo onto a target image at a position defined by normalized YOLO coordinates,
    maintaining the logo's aspect ratio and centering it within the target bounding box.

    Args:
        target_image_path (str): Path to the background image where the logo will be placed.
        logo_path (str): Path to the logo image (can have transparency - alpha channel).
        normalized_bbox (tuple): A tuple (norm_x_center, norm_y_center, norm_width, norm_height)
                                 for logo placement, normalized to 0-1.
        output_image_path (str): Path to save the resulting image with the placed logo.
    """
    try:
        # 1. Load Images
        target_img = cv2.imread(target_image_path)
        logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED to read alpha channel if present

        if target_img is None:
            print(f"Error: Could not load target image from '{target_image_path}'. Skipping placement.")
            return False
        if logo_img is None:
            print(f"Error: Could not load logo image from '{logo_path}'. Skipping placement.")
            return False

        target_h, target_w = target_img.shape[:2]
        logo_original_h, logo_original_w = logo_img.shape[:2]
        logo_aspect_ratio = logo_original_w / logo_original_h

        # 2. Convert Normalized Bbox to Pixel Coordinates for the TARGET PLACEMENT AREA
        norm_x_center, norm_y_center, norm_width, norm_height = normalized_bbox

        target_box_x_center = int(norm_x_center * target_w)
        target_box_y_center = int(norm_y_center * target_h)
        target_box_width = int(norm_width * target_w)
        target_box_height = int(norm_height * target_h)

        # Calculate top-left corner (x1, y1) and bottom-right corner (x2, y2) of the TARGET PLACEMENT AREA
        target_box_x1 = target_box_x_center - target_box_width // 2
        target_box_y1 = target_box_y_center - target_box_height // 2
        target_box_x2 = target_box_x1 + target_box_width
        target_box_y2 = target_box_y1 + target_box_height

        # Ensure target placement area is within image bounds
        target_box_x1 = max(0, target_box_x1)
        target_box_y1 = max(0, target_box_y1)
        target_box_x2 = min(target_w, target_box_x2)
        target_box_y2 = min(target_h, target_box_y2)

        # Adjust target box width/height if clipping occurred
        actual_target_box_width = target_box_x2 - target_box_x1
        actual_target_box_height = target_box_y2 - target_box_y1

        if actual_target_box_width <= 0 or actual_target_box_height <= 0:
            print(f"Warning: Calculated target placement region is invalid (width/height <= 0) for '{os.path.basename(target_image_path)}'. Skipping logo placement.")
            return False

        # 3. Calculate Logo Dimensions preserving aspect ratio to fit within the target box
        # Determine the maximum possible size for the logo while maintaining aspect ratio
        # and fitting within the 'actual_target_box_width' x 'actual_target_box_height' area
        target_box_aspect_ratio = actual_target_box_width / actual_target_box_height

        if logo_aspect_ratio > target_box_aspect_ratio:
            # Logo is relatively wider than the target box, so fit by width
            scaled_logo_width = actual_target_box_width
            scaled_logo_height = int(actual_target_box_width / logo_aspect_ratio)
        else:
            # Logo is relatively taller than the target box, so fit by height
            scaled_logo_height = actual_target_box_height
            scaled_logo_width = int(actual_target_box_height * logo_aspect_ratio)

        # Ensure minimum size for logo after scaling (avoid 0 or very small dimensions)
        if scaled_logo_width <= 0 or scaled_logo_height <= 0:
            print(f"Warning: Scaled logo dimensions became invalid ({scaled_logo_width}x{scaled_logo_height}) for '{os.path.basename(target_image_path)}'. Skipping logo placement.")
            return False

        resized_logo = cv2.resize(logo_img, (scaled_logo_width, scaled_logo_height), interpolation=cv2.INTER_AREA)

        # 4. Calculate Centering Offsets and Final Placement Coordinates
        offset_x = (actual_target_box_width - scaled_logo_width) // 2
        offset_y = (actual_target_box_height - scaled_logo_height) // 2

        # These are the final pixel coordinates where the logo will be placed on the target_img
        logo_final_x1 = target_box_x1 + offset_x
        logo_final_y1 = target_box_y1 + offset_y
        logo_final_x2 = logo_final_x1 + scaled_logo_width
        logo_final_y2 = logo_final_y1 + scaled_logo_height
        
        # Adjusting coordinates to ensure they are within the actual target_img boundary for slicing
        # (This typically handled by max/min above, but a double check never hurts)
        logo_final_x1 = max(0, logo_final_x1)
        logo_final_y1 = max(0, logo_final_y1)
        logo_final_x2 = min(target_w, logo_final_x2)
        logo_final_y2 = min(target_h, logo_final_y2)

        # Re-calculate final dimensions after potential clipping from final bounds check
        final_insertion_width = logo_final_x2 - logo_final_x1
        final_insertion_height = logo_final_y2 - logo_final_y1

        # Adjust resized_logo if final insertion dimensions are slightly different due to integer rounding/clipping
        if final_insertion_width != scaled_logo_width or final_insertion_height != scaled_logo_height:
             resized_logo = cv2.resize(resized_logo, (final_insertion_width, final_insertion_height), interpolation=cv2.INTER_AREA)


        # 5. Overlay Logo onto the precise ROI
        roi = target_img[logo_final_y1:logo_final_y2, logo_final_x1:logo_final_x2]

        if resized_logo.shape[2] == 4: # Logo has an alpha channel (RGBA)
            b, g, r, alpha = cv2.split(resized_logo)
            logo_bgr = cv2.merge([b, g, r])
            alpha_normalized = alpha / 255.0

            # Blend the logo onto the ROI using the alpha channel
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_normalized) + logo_bgr[:, :, c] * alpha_normalized
        else: # Logo has no alpha channel (BGR)
            target_img[logo_final_y1:logo_final_y2, logo_final_x1:logo_final_x2] = resized_logo

        # 6. Save Result
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, target_img)
        return True

    except Exception as e:
        print(f"An error occurred during logo placement on '{os.path.basename(target_image_path)}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes YOLO annotations to find an ideal logo placement, "
                    "then places the logo onto a new set of target images, preserving logo aspect ratio."
    )
    parser.add_argument(
        'annotations_dir',
        type=str,
        help='Path to the directory containing YOLO .txt annotation files (from detection.py).'
    )
    parser.add_argument(
        'logo_image',
        type=str,
        help='Path to the logo image file (e.g., my_logo.png). Supports transparency.'
    )
    parser.add_argument(
        'target_images_for_placement_pattern',
        type=str,
        help='Path pattern to one or more images where the logo will be placed '
             '(e.g., "new_products/*.jpg" or "product1.png product2.jpeg"). '
             'Enclose patterns with wildcards in quotes.'
    )
    parser.add_argument(
        '--placement_output_dir',
        type=str,
        default='placed_logos_output',
        help='Directory to save the images with the logo placed (default: placed_logos_output/).'
    )
    parser.add_argument(
        '--use_median',
        action='store_true',
        help='Use median coordinates for logo placement instead of average. Median is more robust to outliers.'
    )
    parser.add_argument(
        '--reference_width',
        type=int,
        help='Optional: Width of a typical target image to display pixel coordinates during analysis (e.g., 1920).'
    )
    parser.add_argument(
        '--reference_height',
        type=int,
        help='Optional: Height of a typical target image to display pixel coordinates during analysis (e.g., 1080).'
    )
    parser.add_argument(
        '--output_image_suffix',
        type=str,
        default='_with_logo',
        help='Suffix to add to the base filename of the output images (e.g., "_with_logo").'
             ' If empty, original file name is used (overwrites original if in same directory). (default: "_with_logo")'
    )

    args = parser.parse_args()

    print("\n--- Starting Combined Analysis & Logo Placement ---")
    print(f"Annotations Directory for Analysis: '{args.annotations_dir}'")
    print(f"Logo Image: '{args.logo_image}'")
    print(f"Target Images for Placement Pattern: '{args.target_images_for_placement_pattern}'")
    print(f"Output Directory for Placed Images: '{args.placement_output_dir}'")
    print(f"Using {'Median' if args.use_median else 'Average'} for ideal placement.")
    if args.reference_width and args.reference_height:
        print(f"Reference Image Size for Analysis Logging: {args.reference_width}x{args.reference_height}")
    print(f"Output Image Suffix: '{args.output_image_suffix}'")
    print("---------------------------------------------------\n")

    # --- Step 1: Analyze Annotations to find Ideal Bbox ---
    print("--- Analyzing annotations to find ideal placement position ---")
    analysis_results = analyze_yolo_annotations(
        args.annotations_dir,
        args.reference_width,
        args.reference_height
    )

    if analysis_results is None:
        print("\nAnalysis failed or no annotations processed. Aborting logo placement.")
        return

    # Print analysis summary
    print("\n--- Annotation Analysis Summary ---")
    print(f"Total files scanned: {analysis_results['total_files_scanned']}")
    print(f"Total annotations processed: {analysis_results['total_annotations_processed']}")
    if analysis_results['files_with_errors']:
        print(f"Files with errors/warnings during analysis: {', '.join(analysis_results['files_with_errors'])}")
    
    print("\n--- Normalized Bounding Box Statistics ---")
    print("Average:", analysis_results['average_normalized_bbox'])
    print("Median: ", analysis_results['median_normalized_bbox'])
    print("Std Dev:", analysis_results['std_dev_normalized_bbox'])

    if 'average_pixel_bbox_for_reference_image' in analysis_results:
        print(f"\n--- Pixel Bounding Box for {args.reference_width}x{args.reference_height} Reference Image ---")
        print("Average (x1,y1,x2,y2):", (analysis_results['average_pixel_bbox_for_reference_image']['x1'],
                                          analysis_results['average_pixel_bbox_for_reference_image']['y1'],
                                          analysis_results['average_pixel_bbox_for_reference_image']['x2'],
                                          analysis_results['average_pixel_bbox_for_reference_image']['y2']))
        print("Median (x1,y1,x2,y2): ", (analysis_results['median_pixel_bbox_for_reference_image']['x1'],
                                          analysis_results['median_pixel_bbox_for_reference_image']['y1'],
                                          analysis_results['median_pixel_bbox_for_reference_image']['x2'],
                                          analysis_results['median_pixel_bbox_for_reference_image']['y2']))

    # Select the ideal normalized bounding box based on user choice
    if args.use_median:
        ideal_normalized_bbox = analysis_results['median_normalized_bbox']
        print(f"\nUsing Median Normalized Bbox for Placement: {ideal_normalized_bbox}")
    else:
        ideal_normalized_bbox = analysis_results['average_normalized_bbox']
        print(f"\nUsing Average Normalized Bbox for Placement: {ideal_normalized_bbox}")
    
    # Convert ideal_normalized_bbox dict to tuple format (x_c, y_c, w, h) for placement function
    ideal_bbox_tuple = (ideal_normalized_bbox['x_center'],
                        ideal_normalized_bbox['y_center'],
                        ideal_normalized_bbox['width'],
                        ideal_normalized_bbox['height'])


    # --- Step 2: Place Logo on Target Images ---
    print("\n--- Starting Logo Placement on Target Images ---")
    target_image_paths = []
    for part in args.target_images_for_placement_pattern.split():
        target_image_paths.extend(glob.glob(part))

    if not target_image_paths:
        print(f"Error: No target images found for placement matching pattern '{args.target_images_for_placement_pattern}'. Aborting placement.")
        return

    print(f"Found {len(target_image_paths)} images for logo placement.")

    os.makedirs(args.placement_output_dir, exist_ok=True) # Ensure output directory for placed images exists

    for i, target_img_path in enumerate(target_image_paths):
        base_name = os.path.basename(target_img_path)
        name_without_ext, ext = os.path.splitext(base_name)
        output_filename = f"{name_without_ext}{args.output_image_suffix}{ext}"
        output_path = os.path.join(args.placement_output_dir, output_filename)

        print(f"Processing placement on image {i+1}/{len(target_image_paths)}: '{base_name}' -> '{output_filename}'")
        place_logo_on_image(
            target_img_path,
            args.logo_image,
            ideal_bbox_tuple,
            output_path
        )
        print("-" * 30) # Separator

    print("\n--- Combined Process Complete ---")
    print(f"All placed images saved to the '{args.placement_output_dir}' directory.")

if __name__ == "__main__":
    main()
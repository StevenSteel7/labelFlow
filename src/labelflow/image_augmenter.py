# image_augmenter.py (Migrated to PyQt6 - Corrected Syntax)
import os
import random
import cv2
import numpy as np
import json
import traceback # For detailed error logging

# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QMessageBox, QSpinBox,
                             QCheckBox, QDoubleSpinBox, QProgressBar, QApplication,
                             QWidget, QGroupBox, QGridLayout, QScrollArea) # Added QGridLayout, QScrollArea
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- Augmentation Thread ---
class AugmentationThread(QThread):
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # bool: success, str: message/error

    def __init__(self, input_dir, output_dir, options, coco_file=None):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.options = options
        self.coco_file = coco_file
        self.coco_data = None
        self.stop_requested = False

    def run(self):
        try:
            images_output_dir = os.path.join(self.output_dir, "images")
            os.makedirs(images_output_dir, exist_ok=True)
            self.log_message.emit(f"Output images will be saved to: {images_output_dir}")

            image_files = [f for f in os.listdir(self.input_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            if not image_files:
                raise ValueError("No supported image files found in the input directory.")

            num_augmentations = self.options.get("aug_count", 1)
            total_augmentations = len(image_files) * num_augmentations
            self.log_message.emit(f"Found {len(image_files)} images. Performing {num_augmentations} augmentations per image.")

            # Load COCO data if requested
            augmented_coco_data = None
            original_annotations_map = {} # Map image_filename -> list of annotations
            image_id_map = {} # Map image_filename -> original image_id
            if self.coco_file:
                self.log_message.emit(f"Loading COCO annotations from: {self.coco_file}")
                with open(self.coco_file, 'r', encoding='utf-8') as f:
                    self.coco_data = json.load(f)
                # Pre-process annotations for faster lookup
                image_filename_to_id = {img['file_name']: img['id'] for img in self.coco_data['images']}
                for ann in self.coco_data['annotations']:
                    img_id = ann['image_id']
                    img_filename = next((img['file_name'] for img in self.coco_data['images'] if img['id'] == img_id), None)
                    if img_filename:
                        if img_filename not in original_annotations_map:
                            original_annotations_map[img_filename] = []
                        original_annotations_map[img_filename].append(ann)
                        image_id_map[img_filename] = img_id # Store original ID mapping

                augmented_coco_data = {
                    "images": [],
                    "annotations": [],
                    "categories": self.coco_data.get("categories", [])
                }
                self.log_message.emit("COCO data loaded. Annotations will be augmented.")

            processed_count = 0
            next_image_id = 1
            next_annotation_id = 1

            for i, image_file in enumerate(image_files):
                if self.stop_requested:
                    self.log_message.emit("Augmentation stopped by user.")
                    self.finished.emit(False, "Stopped by user.")
                    return

                input_path = os.path.join(self.input_dir, image_file)
                self.log_message.emit(f"Processing: {image_file} ({i+1}/{len(image_files)})")

                try:
                    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        self.log_message.emit(f"Warning: Skipping {image_file} - Failed to load.")
                        # Adjust progress for skipped augmentations
                        processed_count += num_augmentations
                        self.progress.emit(int(processed_count / total_augmentations * 100) if total_augmentations > 0 else 0)
                        continue

                    is_color = len(image.shape) == 3 and image.shape[2] >= 3 # Handle alpha channel too
                    bit_depth = image.dtype

                    original_anns = original_annotations_map.get(image_file, []) if self.coco_file else []

                    for j in range(num_augmentations):
                        if self.stop_requested: break

                        try:
                            augmented_img, transform_params = self._apply_random_augmentation(image)
                            if augmented_img is None: # Check if augmentation failed
                                self.log_message.emit(f"Warning: Augmentation skipped for {image_file} iter {j+1} due to error.")
                                continue

                            # Ensure augmented image matches original properties (color/bit depth)
                            if not is_color and len(augmented_img.shape) == 3:
                                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
                            elif is_color and len(augmented_img.shape) == 2:
                                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_GRAY2BGR)

                            # Ensure bit depth matches
                            if augmented_img.dtype != bit_depth:
                                 if np.issubdtype(bit_depth, np.integer):
                                      info = np.iinfo(bit_depth)
                                      augmented_img = np.clip(augmented_img, info.min, info.max)
                                 elif np.issubdtype(bit_depth, np.floating):
                                      info = np.finfo(bit_depth)
                                      augmented_img = np.clip(augmented_img, info.min, info.max)
                                 try:
                                     augmented_img = augmented_img.astype(bit_depth)
                                 except (ValueError, TypeError) as cast_err:
                                     self.log_message.emit(f"Warning: Could not cast augmented image back to {bit_depth} for {image_file} iter {j+1}. Error: {cast_err}")
                                     continue # Skip this iteration

                            output_filename = f"{os.path.splitext(image_file)[0]}_aug_{j+1}{os.path.splitext(image_file)[1]}"
                            output_path = os.path.join(images_output_dir, output_filename)
                            save_success = cv2.imwrite(output_path, augmented_img)
                            if not save_success:
                                 self.log_message.emit(f"Warning: Failed to save augmented image: {output_path}")
                                 continue # Skip annotation augmentation if image save failed

                            # Augment COCO annotations if enabled
                            if self.coco_file and augmented_coco_data is not None:
                                augmented_coco_data["images"].append({
                                    "id": next_image_id,
                                    "file_name": output_filename,
                                    "height": augmented_img.shape[0],
                                    "width": augmented_img.shape[1]
                                })

                                for ann in original_anns:
                                    augmented_ann = self._augment_annotation(ann, transform_params, augmented_img.shape[:2])
                                    if augmented_ann: # Only add if valid
                                        augmented_ann["id"] = next_annotation_id
                                        augmented_ann["image_id"] = next_image_id
                                        augmented_coco_data["annotations"].append(augmented_ann)
                                        next_annotation_id += 1
                                next_image_id += 1

                        except Exception as aug_err:
                            self.log_message.emit(f"Error augmenting {image_file} (iter {j+1}): {aug_err}")
                            # traceback.print_exc() # Uncomment for detailed debug
                        finally:
                            processed_count += 1
                            self.progress.emit(int(processed_count / total_augmentations * 100) if total_augmentations > 0 else 0)

                except Exception as file_err:
                    self.log_message.emit(f"Error processing file {image_file}: {file_err}")
                    # Adjust progress for skipped augmentations
                    processed_count += num_augmentations
                    self.progress.emit(int(processed_count / total_augmentations * 100) if total_augmentations > 0 else 0)

                if self.stop_requested: break # Break outer loop too

            # Save augmented COCO file if needed
            output_coco_path = ""
            if self.coco_file and augmented_coco_data:
                output_coco_path = os.path.join(self.output_dir, "augmented_annotations.json")
                self.log_message.emit(f"Saving augmented annotations to: {output_coco_path}")
                with open(output_coco_path, 'w', encoding='utf-8') as f:
                    json.dump(augmented_coco_data, f, indent=2)

            self.finished.emit(True, f"Augmentation completed. Augmented annotations saved to {output_coco_path}" if output_coco_path else "Augmentation completed.")

        except Exception as e:
            self.log_message.emit(f"Critical error: {e}")
            traceback.print_exc() # Print stack trace
            self.finished.emit(False, f"Error: {e}")

    def request_stop(self):
        self.stop_requested = True

    # --- Augmentation Functions ---
    def _apply_random_augmentation(self, image):
        """Applies one randomly selected enabled augmentation."""
        enabled_augmentations = []
        opts = self.options # Easier access

        if opts.get("rotate", False): enabled_augmentations.append(self._rotate_image)
        if opts.get("zoom", False): enabled_augmentations.append(self._zoom_image)
        if opts.get("blur", False): enabled_augmentations.append(self._blur_image)
        if opts.get("brightness_contrast", False): enabled_augmentations.append(self._adjust_brightness_contrast)
        if opts.get("sharpen", False): enabled_augmentations.append(self._sharpen_image)
        if opts.get("flip", False): enabled_augmentations.append(self._flip_image)
        # Only add elastic if COCO augmentation is *not* requested
        if opts.get("elastic", False) and not self.coco_file:
            enabled_augmentations.append(self._elastic_transform)
        if opts.get("grayscale", False): enabled_augmentations.append(self._convert_to_grayscale)
        if opts.get("hist_equalize", False): enabled_augmentations.append(self._apply_histogram_equalization)

        if not enabled_augmentations:
            return image.copy(), {"type": "none"} # Return a copy even if no aug applied

        # Select and apply one random augmentation from the enabled list
        aug_func = random.choice(enabled_augmentations)
        try:
            return aug_func(image.copy()) # Apply to a copy
        except Exception as e:
             self.log_message.emit(f"Error during augmentation function {aug_func.__name__}: {e}")
             return None, None # Indicate failure

    # --- Individual Augmentation Methods ---
    # (These methods now take params from self.options and return augmented image + params dict)

    def _rotate_image(self, image):
        max_angle = self.options.get("rotate_angle", 30)
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0) # Positive angle for CCW in OpenCV
        # Use BORDER_REFLECT_101 for smoother edges
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return rotated, {"type": "rotate", "angle": angle, "center": center, "matrix": M}

    def _zoom_image(self, image):
        # Zoom factor: 1.0 means no zoom, > 1 means zoom in, < 1 means zoom out
        max_scale_delta = self.options.get("zoom_factor", 0.2) # Max deviation from 1.0
        min_scale = max(0.1, 1.0 - max_scale_delta) # Ensure scale doesn't go below 0.1
        max_scale = 1.0 + max_scale_delta
        scale = random.uniform(min_scale, max_scale)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        # warpAffine handles both zoom in and out based on the matrix
        zoomed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return zoomed, {"type": "zoom", "scale": scale, "center": center, "matrix": M}

    def _blur_image(self, image):
        kernel_size = random.choice([3, 5, 7]) # Must be odd
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred, {"type": "blur", "kernel_size": kernel_size}

    def _adjust_brightness_contrast(self, image):
        alpha = random.uniform(0.7, 1.3)  # Contrast control (1.0 = no change)
        beta = random.uniform(-30, 30)    # Brightness control (0 = no change)
        # Apply formula: output = image * alpha + beta
        # Need to handle different data types carefully
        if image.dtype == np.uint8:
            adjusted = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            adjusted = np.clip(image * alpha + beta, 0, 65535).astype(np.uint16)
        elif np.issubdtype(image.dtype, np.floating):
            adjusted = (image * alpha + beta).astype(image.dtype) # Assume float range doesn't need clipping here
        else:
            self.log_message.emit(f"Warning: Unsupported dtype {image.dtype} for brightness/contrast. Skipping.")
            return image, {"type": "brightness_contrast", "applied": False}
        return adjusted, {"type": "brightness_contrast", "alpha": alpha, "beta": beta, "applied": True}

    def _sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # Use filter2D. Ensure image is appropriate type (e.g., uint8 or float32)
        if image.dtype == np.uint16:
             # filter2D might not work well directly on uint16, convert temporarily
             img_float = image.astype(np.float32) / 65535.0
             sharpened_float = cv2.filter2D(img_float, -1, kernel)
             sharpened = np.clip(sharpened_float * 65535.0, 0, 65535).astype(np.uint16)
        elif image.dtype == np.uint8 or np.issubdtype(image.dtype, np.floating):
             sharpened = cv2.filter2D(image, -1, kernel) # Works well for uint8/float
        else:
             self.log_message.emit(f"Warning: Unsupported dtype {image.dtype} for sharpening. Skipping.")
             return image, {"type": "sharpen", "applied": False}
        return sharpened, {"type": "sharpen", "applied": True}

    def _flip_image(self, image):
        flip_codes = self.options.get("flip_codes", []) # Get pre-calculated list
        if not flip_codes:
            return image, {"type": "flip", "flip_code": None}
        flip_code = random.choice(flip_codes)
        flipped = cv2.flip(image, flip_code)
        return flipped, {"type": "flip", "flip_code": flip_code}

    def _elastic_transform(self, image):
        alpha = self.options.get("elastic_alpha", 300)
        sigma = self.options.get("elastic_sigma", 15)
        shape = image.shape[:2] # Use only H, W for displacement map
        random_state = np.random.RandomState(None)

        # Create float32 displacement fields
        dx = (random_state.rand(*shape) * 2 - 1).astype(np.float32)
        dy = (random_state.rand(*shape) * 2 - 1).astype(np.float32)

        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # Create coordinate grid and apply displacement
        x, y = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
        map_x = x + dx
        map_y = y + dy

        # Remap the image
        transformed = cv2.remap(image, map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

        # Return image and transformation parameters (though dx/dy aren't easily reversible for annotations)
        return transformed, {"type": "elastic"} # Annotation augmentation not supported here

    def _convert_to_grayscale(self, image):
        if len(image.shape) == 2: return image, {"type": "grayscale", "applied": True} # Already grayscale
        if len(image.shape) == 3 and image.shape[2] == 1: return image, {"type": "grayscale", "applied": True} # Also effectively grayscale

        if len(image.shape) == 3 and image.shape[2] == 4: # Handle BGRA
             gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 3: # Handle BGR
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
             self.log_message.emit(f"Grayscale conversion skipped: Unsupported image shape {image.shape}")
             return image, {"type": "grayscale", "applied": False}

        # Keep grayscale as 2D array
        return gray, {"type": "grayscale", "applied": True}

    def _apply_histogram_equalization(self, image):
         # --- Helper for 16-bit ---
        def equalize_16bit(img_16):
            hist, bins = np.histogram(img_16.flatten(), 65536, [0, 65536])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0) # Mask zero values in CDF
            # Handle empty mask case (flat image)
            if cdf_m.mask.all(): return img_16
            # Calculate normalized CDF
            cdf_min = cdf_m.min() # Use min of non-masked values
            cdf_max = cdf_m.max() # Use max of non-masked values
            if cdf_max == cdf_min: return img_16 # Avoid division by zero if range is zero
            cdf_m = (cdf_m - cdf_min) * 65535 / (cdf_max - cdf_min)
            cdf_final = np.ma.filled(cdf_m, 0).astype('uint16') # Fill masked values (0) back
            return cdf_final[img_16] # Apply mapping

        if len(image.shape) == 3 and image.shape[2] >= 3: # Color image (BGR or BGRA)
             img_yuv = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2YUV) # Work on first 3 channels
             y_channel = img_yuv[:,:,0]
             if y_channel.dtype == np.uint8:
                  img_yuv[:,:,0] = cv2.equalizeHist(y_channel)
             elif y_channel.dtype == np.uint16:
                  img_yuv[:,:,0] = equalize_16bit(y_channel)
             else:
                  print(f"Warning: Unsupported dtype {y_channel.dtype} for histogram equalization.")
                  return image, {"type": "histogram_equalization", "applied": False}
             equalized_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
             # If original had alpha, merge it back
             if image.shape[2] == 4:
                 equalized = cv2.merge((equalized_bgr, image[:,:,3]))
             else:
                 equalized = equalized_bgr
             return equalized, {"type": "histogram_equalization", "applied": True}
        elif len(image.shape) == 2: # Grayscale
             if image.dtype == np.uint8:
                  equalized = cv2.equalizeHist(image)
             elif image.dtype == np.uint16:
                  equalized = equalize_16bit(image)
             else:
                  print(f"Warning: Unsupported dtype {image.dtype} for histogram equalization.")
                  return image, {"type": "histogram_equalization", "applied": False}
             return equalized, {"type": "histogram_equalization", "applied": True}
        else: # Unsupported shape
             self.log_message.emit(f"Histogram equalization skipped: Unsupported image shape {image.shape}")
             return image, {"type": "histogram_equalization", "applied": False}

    # --- Annotation Augmentation ---
    def _augment_annotation(self, annotation, transform_params, original_shape):
        """Applies the geometric transformation to a single COCO annotation."""
        augmented_ann = annotation.copy() # Start with a copy
        transform_type = transform_params.get("type", "none")

        # We only modify segmentation/bbox if a geometric augmentation happened
        if transform_type not in ["rotate", "zoom", "flip"]:
            return augmented_ann # Return original if no geometric transform

        has_segmentation = "segmentation" in augmented_ann and augmented_ann["segmentation"]
        has_bbox = "bbox" in augmented_ann and len(augmented_ann["bbox"]) == 4

        new_segmentation_flat = None
        transformed_points = None

        # --- Transform Segmentation Points ---
        if has_segmentation:
            # COCO segmentation can be list of lists [[poly1], [poly2]...]
            # We typically augment only the first polygon for simplicity here
            seg = augmented_ann["segmentation"][0] # Assume first part is the main one
            if not isinstance(seg, list) or len(seg) < 6:
                self.log_message.emit(f"Warning: Invalid segmentation format for ann ID {annotation.get('id', 'N/A')}. Skipping augmentation.")
                return None # Invalid format

            points = np.array(seg).reshape(-1, 2).astype(np.float32)
            transformed_points = None

            if transform_type == "rotate":
                 matrix = transform_params["matrix"]
                 transformed_points = self._transform_points(points, matrix)
            elif transform_type == "zoom":
                 matrix = transform_params["matrix"]
                 transformed_points = self._transform_points(points, matrix)
            elif transform_type == "flip":
                 flip_code = transform_params["flip_code"]
                 if flip_code is not None:
                      transformed_points = self._flip_points(points, flip_code, original_shape)
                 else:
                      transformed_points = points # No flip applied

            if transformed_points is not None:
                # Clip coordinates to image bounds
                aug_h, aug_w = original_shape # Assumes output image size matches input
                transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, aug_w - 1)
                transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, aug_h - 1)

                # Filter out points that might become identical after clipping/transformation
                # and ensure closed polygon if needed (OpenCV findContours handles this better)
                unique_points_list = []
                if len(transformed_points) > 0:
                     unique_points_list.append(transformed_points[0])
                     for i in range(1, len(transformed_points)):
                          # Use a small tolerance for comparing floating point coords
                          if not np.allclose(transformed_points[i], transformed_points[i-1], atol=1e-2):
                               unique_points_list.append(transformed_points[i])

                # If the first and last point are now identical, remove the last one
                if len(unique_points_list) > 1 and np.allclose(unique_points_list[0], unique_points_list[-1], atol=1e-2):
                     unique_points_list.pop()

                # Check if enough points remain for a valid polygon
                if len(unique_points_list) < 3:
                    self.log_message.emit(f"Warning: Polygon for ann ID {annotation.get('id', 'N/A')} became invalid after transform. Skipping.")
                    return None # Polygon became invalid

                new_segmentation_flat = np.array(unique_points_list).flatten().tolist()
                augmented_ann["segmentation"] = [new_segmentation_flat] # Store back in COCO format

        # --- Handle BBox ---
        # Always recalculate BBox from segmentation if segmentation was processed
        if new_segmentation_flat:
            new_bbox = self._get_bbox_from_polygon(new_segmentation_flat)
            if new_bbox:
                 augmented_ann["bbox"] = new_bbox
            else: # If polygon somehow resulted in invalid bbox
                 augmented_ann["bbox"] = [] # Clear bbox
                 self.log_message.emit(f"Warning: Could not calculate valid BBox for ann ID {annotation.get('id', 'N/A')} after transform.")

        elif has_bbox and not has_segmentation:
            # If only bbox existed originally, transform its corners
            x, y, w, h = augmented_ann["bbox"]
            # Define corners: top-left, top-right, bottom-right, bottom-left
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            transformed_corners = None

            if transform_type == "rotate":
                 matrix = transform_params["matrix"]
                 transformed_corners = self._transform_points(corners, matrix)
            elif transform_type == "zoom":
                 matrix = transform_params["matrix"]
                 transformed_corners = self._transform_points(corners, matrix)
            elif transform_type == "flip":
                 flip_code = transform_params["flip_code"]
                 if flip_code is not None:
                      transformed_corners = self._flip_points(corners, flip_code, original_shape)
                 else:
                      transformed_corners = corners # No flip applied

            if transformed_corners is not None:
                new_bbox = self._get_bbox_from_points(transformed_corners) # Calc bbox from transformed corners
                if new_bbox:
                     augmented_ann["bbox"] = new_bbox
                else:
                     augmented_ann["bbox"] = [] # Clear if invalid
                     self.log_message.emit(f"Warning: BBox for ann ID {annotation.get('id', 'N/A')} became invalid after transform.")

        # --- Recalculate Area ---
        # Always recalculate area if bbox exists, otherwise set to 0
        if "bbox" in augmented_ann and augmented_ann["bbox"]:
             augmented_ann["area"] = float(augmented_ann["bbox"][2] * augmented_ann["bbox"][3])
        else:
             augmented_ann["area"] = 0.0

        return augmented_ann

    def _transform_points(self, points, matrix):
        """Applies an affine transformation matrix to points."""
        # points is Nx2 array, matrix is 2x3
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))]) # Nx3
        transformed_points = matrix.dot(points_homogeneous.T).T # (2x3 * 3xN).T -> Nx2
        return transformed_points

    def _flip_points(self, points, flip_code, image_shape):
        """Flips points based on OpenCV flip code."""
        h, w = image_shape[:2]
        flipped_points = points.copy()
        if flip_code == 0:  # Vertical flip (around x-axis)
            flipped_points[:, 1] = h - 1 - points[:, 1]
        elif flip_code == 1:  # Horizontal flip (around y-axis)
            flipped_points[:, 0] = w - 1 - points[:, 0]
        elif flip_code == -1:  # Both
            flipped_points[:, 0] = w - 1 - points[:, 0]
            flipped_points[:, 1] = h - 1 - points[:, 1]
        return flipped_points

    def _get_bbox_from_points(self, points):
        """Calculates the bounding box [x, y, w, h] from points."""
        if points is None or points.shape[0] == 0: return None
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0: return None # Invalid bbox
        return [float(x_min), float(y_min), float(w), float(h)]

    def _get_bbox_from_polygon(self, polygon_flat):
        """Calculates bbox [x,y,w,h] from a flat list polygon."""
        if not polygon_flat or len(polygon_flat) < 6: return None
        points = np.array(polygon_flat).reshape(-1, 2)
        return self._get_bbox_from_points(points)

    def _calculate_polygon_area(self, polygon_flat):
        """Calculates area of a polygon from a flat list."""
        if not polygon_flat or len(polygon_flat) < 6: return 0.0
        points = np.array(polygon_flat).reshape(-1, 2)
        # Shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        # Ensure closing point is not duplicated for calculation
        if np.allclose(points[0], points[-1]):
            x = x[:-1]
            y = y[:-1]
        if len(x) < 3: return 0.0 # Need at least 3 distinct points
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --- Main Dialog ---
class ImageAugmenterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Augmenter")
        self.setGeometry(100, 100, 500, 700) # Increased height further
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window) # Use enum
        self.setWindowModality(Qt.WindowModality.ApplicationModal) # Use enum
        self.input_dir = ""
        self.output_dir = ""
        self.coco_file = ""
        self.augmentation_thread = None
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15) # Added spacing

        # --- File/Directory Selection ---
        io_group = QGroupBox("Input / Output")
        io_layout = QGridLayout()
        io_layout.setColumnStretch(1, 1) # Allow label to stretch
        io_layout.setSpacing(8)

        self.input_label = QLabel("Not selected")
        self.input_label.setWordWrap(True)
        input_button = QPushButton("...")
        input_button.setFixedSize(30, 25) # Small browse button
        input_button.setToolTip("Select Input Image Directory")
        input_button.clicked.connect(self.select_input_directory)
        io_layout.addWidget(QLabel("Input Dir:"), 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop) # Align label
        io_layout.addWidget(self.input_label, 0, 1)
        io_layout.addWidget(input_button, 0, 2)

        self.output_label = QLabel("Not selected")
        self.output_label.setWordWrap(True)
        output_button = QPushButton("...")
        output_button.setFixedSize(30, 25)
        output_button.setToolTip("Select Output Directory")
        output_button.clicked.connect(self.select_output_directory)
        io_layout.addWidget(QLabel("Output Dir:"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop) # Align label
        io_layout.addWidget(self.output_label, 1, 1)
        io_layout.addWidget(output_button, 1, 2)

        self.coco_check = QCheckBox("Augment COCO Annotations")
        self.coco_check.setToolTip("If checked, selects a COCO JSON file and augments annotations along with images.")
        self.coco_check.stateChanged.connect(self.toggle_coco_options)
        io_layout.addWidget(self.coco_check, 2, 0, 1, 3) # Span all columns

        self.coco_label = QLabel("Not selected")
        self.coco_label.setWordWrap(True)
        self.coco_label.setVisible(False) # Hide initially
        self.coco_button = QPushButton("...")
        self.coco_button.setFixedSize(30, 25)
        self.coco_button.setToolTip("Select COCO JSON Annotation File")
        self.coco_button.clicked.connect(self.select_coco_json)
        self.coco_button.setVisible(False) # Hide initially
        io_layout.addWidget(QLabel("COCO JSON:"), 3, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop) # Align label
        io_layout.addWidget(self.coco_label, 3, 1)
        io_layout.addWidget(self.coco_button, 3, 2)

        io_group.setLayout(io_layout)
        self.main_layout.addWidget(io_group)

        # --- Augmentation Options ---
        options_group = QGroupBox("Augmentation Options")
        options_layout = QVBoxLayout()

        # Number of augmentations per image
        aug_count_layout = QHBoxLayout()
        aug_count_layout.addWidget(QLabel("Augmentations per image:"))
        self.aug_count_spin = QSpinBox()
        self.aug_count_spin.setRange(1, 100)
        self.aug_count_spin.setValue(5)
        aug_count_layout.addWidget(self.aug_count_spin)
        aug_count_layout.addStretch()
        options_layout.addLayout(aug_count_layout)
        options_layout.addSpacing(10)

        # ScrollArea for transformations
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(300) # Fixed height for scroll area
        scroll_content = QWidget()
        transforms_layout = QVBoxLayout(scroll_content)
        transforms_layout.setSpacing(8) # Spacing between transforms

        # --- Transformations ---
        transforms_layout.addWidget(QLabel("<b>Enabled Transformations:</b>"))

        # Rotate
        self.rotate_check = QCheckBox("Rotate")
        self.rotate_spin = QSpinBox()
        self.rotate_spin.setRange(1, 180) # Use positive range for max deviation
        self.rotate_spin.setValue(30)
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(self.rotate_check)
        rotate_layout.addWidget(QLabel("Max Angle (°):"))
        rotate_layout.addWidget(self.rotate_spin)
        rotate_layout.addStretch()
        transforms_layout.addLayout(rotate_layout)

        # Zoom
        self.zoom_check = QCheckBox("Zoom")
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.0, 1.0) # Factor range (0=no change, 1=up to 2x or 0x -> use 0.0 to 1.0 for delta)
        self.zoom_spin.setValue(0.2) # e.g., 0.2 means scale between 0.8 and 1.2
        self.zoom_spin.setSingleStep(0.05)
        self.zoom_spin.setDecimals(2)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_check)
        zoom_layout.addWidget(QLabel("Max Scale Δ (+/-):")) # Clarified label
        zoom_layout.addWidget(self.zoom_spin)
        zoom_layout.addStretch()
        transforms_layout.addLayout(zoom_layout)

        # Flip
        self.flip_check = QCheckBox("Flip")
        self.flip_check.setEnabled(False) # Controlled by sub-checkboxes
        self.flip_horizontal_check = QCheckBox("Horizontal")
        self.flip_vertical_check = QCheckBox("Vertical")
        self.flip_horizontal_check.setChecked(True) # Default H flip
        self.flip_horizontal_check.stateChanged.connect(self.update_flip_check)
        self.flip_vertical_check.stateChanged.connect(self.update_flip_check)
        flip_layout = QHBoxLayout()
        flip_layout.addWidget(self.flip_check)
        flip_layout.addWidget(self.flip_horizontal_check)
        flip_layout.addWidget(self.flip_vertical_check)
        flip_layout.addStretch()
        transforms_layout.addLayout(flip_layout)
        self.update_flip_check() # Initial update

        # Blur
        self.blur_check = QCheckBox("Gaussian Blur")
        transforms_layout.addWidget(self.blur_check)

        # Brightness/Contrast
        self.brightness_contrast_check = QCheckBox("Random Brightness/Contrast")
        transforms_layout.addWidget(self.brightness_contrast_check)

        # Sharpen
        self.sharpen_check = QCheckBox("Sharpen")
        transforms_layout.addWidget(self.sharpen_check)

        # Elastic Deformation (Disabled if COCO is checked)
        self.elastic_check = QCheckBox("Elastic Deformation")
        self.elastic_check.setToolTip("Note: Cannot be used when augmenting COCO annotations.")
        elastic_layout = QHBoxLayout()
        elastic_layout.addWidget(self.elastic_check)
        elastic_layout.addWidget(QLabel("Alpha:"))
        self.elastic_alpha_spin = QSpinBox()
        self.elastic_alpha_spin.setRange(1, 1000)
        self.elastic_alpha_spin.setValue(300) # Adjusted default
        elastic_layout.addWidget(self.elastic_alpha_spin)
        elastic_layout.addWidget(QLabel("Sigma:"))
        self.elastic_sigma_spin = QSpinBox()
        self.elastic_sigma_spin.setRange(1, 100)
        self.elastic_sigma_spin.setValue(15) # Adjusted default
        elastic_layout.addWidget(self.elastic_sigma_spin)
        elastic_layout.addStretch()
        transforms_layout.addLayout(elastic_layout)

        # Grayscale Conversion
        self.grayscale_check = QCheckBox("Convert to Grayscale")
        transforms_layout.addWidget(self.grayscale_check)

        # Histogram Equalization
        self.hist_equalize_check = QCheckBox("Histogram Equalization")
        self.hist_equalize_check.setToolTip("Applies histogram equalization. Works best on 8-bit images.")
        transforms_layout.addWidget(self.hist_equalize_check)

        # --- End Transformations ---
        scroll.setWidget(scroll_content)
        options_layout.addWidget(scroll)

        options_group.setLayout(options_layout)
        self.main_layout.addWidget(options_group)

        # --- Action Buttons & Progress ---
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Augmentation")
        self.start_button.clicked.connect(self.start_augmentation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_augmentation)
        self.stop_button.setEnabled(False)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Idle") # Add a status label
        self.status_label.setStyleSheet("color: gray;")
        self.main_layout.addWidget(self.status_label)

        # Initial state update
        self.toggle_coco_options()

    def toggle_coco_options(self):
        """Enable/disable COCO file selection based on checkbox."""
        checked = self.coco_check.isChecked()
        self.coco_label.setVisible(checked)
        self.coco_button.setVisible(checked)
        # Disable elastic deformation if COCO is checked
        self.elastic_check.setEnabled(not checked)
        if checked:
             self.elastic_check.setChecked(False) # Uncheck it if disabled

    def update_flip_check(self):
        """Update the main Flip checkbox based on sub-options."""
        self.flip_check.setChecked(self.flip_horizontal_check.isChecked() or self.flip_vertical_check.isChecked())

    def select_input_directory(self):
        """Select the input image directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Image Directory")
        if directory:
            self.input_dir = directory
            self.input_label.setText(f"...{os.sep}{os.path.basename(directory)}")
            self.input_label.setToolTip(directory)

    def select_output_directory(self):
        """Select the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"...{os.sep}{os.path.basename(directory)}")
            self.output_label.setToolTip(directory)

    def select_coco_json(self):
        """Select the COCO annotation file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON Annotation File", "", "JSON Files (*.json)")
        if file_name:
            self.coco_file = file_name
            self.coco_label.setText(f"...{os.sep}{os.path.basename(file_name)}")
            self.coco_label.setToolTip(file_name)
            self.coco_check.setChecked(True) # Ensure parent checkbox is checked
        else:
             # If selection is cancelled, clear the file path
             self.coco_file = ""
             self.coco_label.setText("Not selected")
             self.coco_label.setToolTip("")
             # Optionally uncheck the main box if no file is selected
             # self.coco_check.setChecked(False)

    def get_augmentation_options(self):
        """Collects selected augmentation options from the UI."""
        flip_codes = []
        if self.flip_horizontal_check.isChecked(): flip_codes.append(1)
        if self.flip_vertical_check.isChecked(): flip_codes.append(0)
        if self.flip_horizontal_check.isChecked() and self.flip_vertical_check.isChecked(): flip_codes.append(-1)

        return {
            "aug_count": self.aug_count_spin.value(),
            "rotate": self.rotate_check.isChecked(),
            "rotate_angle": self.rotate_spin.value(),
            "zoom": self.zoom_check.isChecked(),
            "zoom_factor": self.zoom_spin.value(),
            "blur": self.blur_check.isChecked(),
            "brightness_contrast": self.brightness_contrast_check.isChecked(),
            "sharpen": self.sharpen_check.isChecked(),
            "flip": self.flip_check.isChecked() and flip_codes, # Only true if sub-option selected
            "flip_codes": flip_codes, # Store actual codes to use
            "elastic": self.elastic_check.isChecked() and self.elastic_check.isEnabled(), # Check if enabled too
            "elastic_alpha": self.elastic_alpha_spin.value(),
            "elastic_sigma": self.elastic_sigma_spin.value(),
            "grayscale": self.grayscale_check.isChecked(),
            "hist_equalize": self.hist_equalize_check.isChecked(),
        }

    def start_augmentation(self):
        """Starts the augmentation process in a separate thread."""
        if not self.input_dir or not os.path.isdir(self.input_dir):
            QMessageBox.warning(self, "Input Directory Error", "Please select a valid input directory.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Output Directory Error", "Please select an output directory.")
            return
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output dir exists

        coco_file_to_use = self.coco_file if self.coco_check.isChecked() else None
        if self.coco_check.isChecked() and not coco_file_to_use:
            QMessageBox.warning(self, "COCO File Error", "Annotation augmentation checked, but no COCO file selected.")
            return

        # Prevent starting if already running
        if self.augmentation_thread and self.augmentation_thread.isRunning():
            QMessageBox.information(self, "Busy", "Augmentation is already in progress.")
            return

        options = self.get_augmentation_options()
        if not any(options.get(k, False) for k in options if k != "aug_count"): # Check if any transform is enabled
             QMessageBox.information(self, "No Augmentations Selected", "Please select at least one augmentation type.")
             return

        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Starting...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Create and start the thread
        self.augmentation_thread = AugmentationThread(
            self.input_dir, self.output_dir, options, coco_file_to_use
        )
        self.augmentation_thread.progress.connect(self.progress_bar.setValue)
        self.augmentation_thread.log_message.connect(self.log) # Add logging later if needed
        self.augmentation_thread.finished.connect(self.on_augmentation_finished)
        self.augmentation_thread.start()

    def stop_augmentation(self):
        """Requests the augmentation thread to stop."""
        if self.augmentation_thread and self.augmentation_thread.isRunning():
            self.augmentation_thread.request_stop()
            self.stop_button.setEnabled(False) # Disable stop button after request
            self.start_button.setText("Stopping...") # Indicate stopping state

    def on_augmentation_finished(self, success, message):
        """Handles the signal emitted when the thread finishes."""
        self.progress_bar.setValue(100 if success else self.progress_bar.value()) # Max out on success
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.start_button.setText("Start Augmentation") # Reset button text

        if success:
            QMessageBox.information(self, "Augmentation Complete", message)
        else:
            QMessageBox.warning(self, "Augmentation Stopped or Failed", message)
        self.status_label.setText(f"Status: {message}")
        self.augmentation_thread = None # Clear the thread reference

    def log(self, message):
        """Updates the status label."""
        # Keep only the last status message to avoid excessive length
        self.status_label.setText(f"Status: {message}")
        print(message) # Optionally print detailed logs to console

    def closeEvent(self, event):
        """Handle closing the dialog while augmentation might be running."""
        if self.augmentation_thread and self.augmentation_thread.isRunning():
             reply = QMessageBox.question(self, "Augmentation Running",
                                          "Augmentation is in progress. Are you sure you want to stop and close?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Use enum
                                          QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes: # Use enum
                 self.stop_augmentation()
                 # We don't wait here, allow closing. Thread should stop cleanly.
                 event.accept()
             else:
                 event.ignore()
        else:
             event.accept()

    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            if self.rect().isValid(): # Check if self geometry is valid
                self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents() # Ensure UI updates


# Function to show the dialog
def show_image_augmenter(parent):
    dialog = ImageAugmenterDialog(parent)
    dialog.show_centered(parent)
    return dialog # Return instance if needed
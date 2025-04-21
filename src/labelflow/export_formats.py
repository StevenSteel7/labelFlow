# export_formats.py (Migrated to PyQt6)
import json
# PyQt6 Import
from PyQt6.QtGui import QImage
from .utils import calculate_area, calculate_bbox
import yaml
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

import numpy as np
import skimage.draw
from PIL import Image
import traceback # For detailed error logging

# Utility function remains internal to this module if needed, but create_coco_annotation is preferred
# def convert_to_coco(...):
#     pass

def create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping):
    """Creates a COCO annotation dictionary from the internal format."""
    if class_name not in class_mapping:
         print(f"Warning: Class '{class_name}' not found in class_mapping. Skipping annotation ID {annotation_id}.")
         return None # Skip if class is invalid

    coco_ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_mapping[class_name],
        "area": 0.0, # Default area
        "iscrowd": 0,
        "bbox": [] # Default bbox
    }

    segmentation = ann.get("segmentation")
    bbox = ann.get("bbox")

    if segmentation:
         # Ensure segmentation is a flat list of numbers
         if isinstance(segmentation, list) and segmentation:
              if isinstance(segmentation[0], list): # Handle potential nested list [[coords]]
                   flat_segmentation = segmentation[0]
              else:
                   flat_segmentation = segmentation

              if len(flat_segmentation) >= 6: # Need at least 3 points
                   # COCO expects list of lists format [[x1,y1,x2,y2,...]]
                   coco_ann["segmentation"] = [flat_segmentation]
                   # Calculate area and bbox *only if* segmentation is valid
                   try:
                        coco_ann["area"] = calculate_area(ann) # Assumes ann format is handled by calculate_area
                        coco_ann["bbox"] = calculate_bbox(flat_segmentation) # Use flat list for bbox calc
                   except Exception as e:
                        print(f"Warning: Could not calculate area/bbox for annotation ID {annotation_id}: {e}")
                        coco_ann["segmentation"] = [] # Clear invalid segmentation
                        coco_ann["bbox"] = []
                        coco_ann["area"] = 0.0
              else:
                   # Invalid segmentation, clear it
                   coco_ann["segmentation"] = []
                   print(f"Warning: Invalid segmentation (less than 3 points) for annotation ID {annotation_id}. Segmentation cleared.")
                   # Try to use bbox if available
                   if bbox and len(bbox) == 4:
                       coco_ann["bbox"] = list(map(float, bbox))
                       coco_ann["area"] = float(bbox[2]) * float(bbox[3])
                   else:
                       coco_ann["bbox"] = []
                       coco_ann["area"] = 0.0

         else: # Invalid segmentation type or empty list
             coco_ann["segmentation"] = []
             print(f"Warning: Invalid segmentation data type or empty list for annotation ID {annotation_id}. Segmentation cleared.")
             # Try to use bbox if available
             if bbox and len(bbox) == 4:
                 coco_ann["bbox"] = list(map(float, bbox))
                 coco_ann["area"] = float(bbox[2]) * float(bbox[3])
             else:
                 coco_ann["bbox"] = []
                 coco_ann["area"] = 0.0

    elif bbox and len(bbox) == 4:
        # If only bbox exists, COCO usually doesn't have segmentation
        coco_ann["segmentation"] = []
        coco_ann["bbox"] = list(map(float, bbox))
        coco_ann["area"] = float(bbox[2]) * float(bbox[3])
    else:
        # No valid segmentation or bbox
        print(f"Warning: No valid segmentation or bbox for annotation ID {annotation_id}. Area and Bbox set to defaults.")
        coco_ann["segmentation"] = []
        coco_ann["bbox"] = []
        coco_ann["area"] = 0.0

    return coco_ann


def export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, json_filename=None):
    """Exports annotations to COCO JSON format."""
    print("Starting COCO JSON export...")
    coco_format = {
        "images": [],
        "categories": [{"id": id, "name": name, "supercategory": ""} for name, id in class_mapping.items()], # Added supercategory
        "annotations": []
    }

    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    print(f"Ensured images directory exists: {images_dir}")

    annotation_id = 1
    image_id = 1
    # Create a mapping of slice names to their QImage objects for quick lookup
    slice_map = {name: img for name, img in slices}
    processed_image_files = set() # Keep track of images already added

    # Iterate through all potential image/slice names present in annotations
    annotated_keys = list(all_annotations.keys())
    print(f"Processing {len(annotated_keys)} annotated image/slice keys...")

    for image_name in annotated_keys:
        annotations = all_annotations.get(image_name)
        # Skip if there are no actual annotations for this key, even if the key exists
        if not annotations or not any(annotations.values()):
            # print(f"Skipping {image_name}: No annotations.")
            continue

        qimage = None
        source_path = None
        is_slice = False
        target_filename = None # The filename to use in the COCO JSON and for saving/copying

        # Determine if it's a slice or an original image file
        if image_name in slice_map:
             is_slice = True
             qimage = slice_map[image_name]
             target_filename = f"{image_name}.png" # Slices are saved as PNG
             print(f"Processing slice: {image_name} -> {target_filename}")
        elif '_' in image_name and '.' not in image_name: # Heuristic for other slices (e.g., CZI/TIFF)
             # Find the corresponding QImage in the main slices list or image_slices dict
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice:
                  qimage = matching_slice[1]
             else:
                  for stack_slices in image_slices.values():
                      matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                      if matching_stack_slice:
                           qimage = matching_stack_slice[1]
                           break
             if qimage:
                  is_slice = True
                  target_filename = f"{image_name}.png"
                  print(f"Processing derived slice: {image_name} -> {target_filename}")
             else:
                  print(f"Warning: No image data found for potential slice key '{image_name}', skipping.")
                  continue
        else:
             # Assume it's an original image file name from image_paths
             source_path = image_paths.get(image_name)
             if not source_path or not os.path.exists(source_path):
                  print(f"Warning: Source path not found or invalid for image key '{image_name}', skipping.")
                  continue
             # Skip raw multi-page files themselves, only process their slices if annotated
             if source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                  print(f"Skipping multi-page source file entry: {image_name}. Slices handled separately.")
                  continue
             target_filename = image_name # Use original filename
             # Load QImage to get dimensions
             qimage = QImage(source_path)
             print(f"Processing original image file: {image_name}")


        # Ensure we have a valid QImage and target filename
        if qimage is None or qimage.isNull() or target_filename is None:
             print(f"Warning: Could not load or determine filename for '{image_name}', skipping.")
             continue

        # Skip if this image file has already been added to COCO images
        if target_filename in processed_image_files:
            # Find existing image ID to add annotations
            existing_image_info = next((img for img in coco_format["images"] if img["file_name"] == target_filename), None)
            if existing_image_info:
                current_image_id = existing_image_info["id"]
                print(f"Adding annotations to existing image: {target_filename} (ID: {current_image_id})")
            else:
                # This case should ideally not happen if processed_image_files is managed correctly
                print(f"Error: Image {target_filename} marked processed but not found in COCO data. Skipping annotations.")
                continue
        else:
            # Add new image entry to COCO data
            image_info = {
                "file_name": target_filename,
                "height": qimage.height(),
                "width": qimage.width(),
                "id": image_id
            }
            coco_format["images"].append(image_info)
            processed_image_files.add(target_filename)
            current_image_id = image_id
            print(f"Added new image to COCO: {target_filename} (ID: {current_image_id})")

            # Save/Copy the image file to the output images directory
            save_path = os.path.join(images_dir, target_filename)
            if not os.path.exists(save_path):
                 if is_slice:
                      qimage.save(save_path)
                      print(f"Saved slice image to: {save_path}")
                 elif source_path:
                      shutil.copy2(source_path, save_path)
                      print(f"Copied original image to: {save_path}")
            # else:
                 # print(f"Image {target_filename} already exists in output directory.")

            image_id += 1 # Increment image ID only for new image entries


        # Add annotations for this image/slice
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                coco_ann = create_coco_annotation(ann, current_image_id, annotation_id, class_name, class_mapping)
                if coco_ann: # Only add if valid
                    coco_format["annotations"].append(coco_ann)
                    annotation_id += 1

    # Generate JSON filename if not provided
    if json_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"annotations_{timestamp}.json"
    elif not json_filename.lower().endswith('.json'):
        json_filename += '.json'

    # Save COCO JSON file
    json_file_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f: # Specify encoding
            json.dump(coco_format, f, indent=2)
        print(f"Successfully exported COCO JSON to: {json_file_path}")
    except Exception as e:
        print(f"Error writing COCO JSON file: {e}")
        raise # Re-raise the exception

    return json_file_path, images_dir


def export_yolo_v4(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports annotations in YOLO v4 format."""
    print("Starting YOLO v4 export...")
    # Create output directories (YOLO v4 typically uses train/valid at top level)
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid') # Create valid even if not used for split
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
    print(f"Ensured YOLO v4 directories exist: {train_dir}, {valid_dir}")

    # Create a mapping of class names to YOLO indices
    class_to_index = {name: i for i, name in enumerate(class_mapping.keys())}

    # Create a mapping of slice names to their QImage objects
    slice_map = {name: img for name, img in slices}

    processed_files = 0
    skipped_files = 0

    for image_name, annotations in all_annotations.items():
        if not annotations or not any(annotations.values()):
            continue # Skip keys with no actual annotations

        # Determine image source and dimensions (similar logic to COCO export)
        qimage = None
        source_path = None
        is_slice = False
        target_filename_base = None
        img_width, img_height = 0, 0

        if image_name in slice_map:
             is_slice = True
             qimage = slice_map[image_name]
             target_filename_base = image_name # Base name for .png and .txt
        elif '_' in image_name and '.' not in image_name:
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice: qimage = matching_slice[1]
             else:
                 for stack_slices in image_slices.values():
                     matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                     if matching_stack_slice: qimage = matching_stack_slice[1]; break
             if qimage: is_slice = True; target_filename_base = image_name
        else:
             source_path = image_paths.get(image_name)
             if source_path and os.path.exists(source_path) and not source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                 target_filename_base = os.path.splitext(image_name)[0]
                 qimage = QImage(source_path) # Load to get dimensions

        if qimage is None or qimage.isNull():
            # print(f"Warning: Skipping '{image_name}', could not load image data.")
            skipped_files += 1
            continue

        img_width, img_height = qimage.width(), qimage.height()
        if img_width <= 0 or img_height <= 0:
             print(f"Warning: Skipping '{image_name}', invalid dimensions ({img_width}x{img_height}).")
             skipped_files += 1
             continue

        # For simplicity, put all data in the train directory
        # In practice, you might implement a train/val split here
        target_img_dir = os.path.join(train_dir, 'images')
        target_lbl_dir = os.path.join(train_dir, 'labels')
        image_file_ext = ".png" if is_slice else os.path.splitext(image_name)[1]
        target_img_filename = f"{target_filename_base}{image_file_ext}"
        target_lbl_filename = f"{target_filename_base}.txt"
        target_img_path = os.path.join(target_img_dir, target_img_filename)
        target_lbl_path = os.path.join(target_lbl_dir, target_lbl_filename)

        # Save/Copy image file if it doesn't exist
        if not os.path.exists(target_img_path):
             if is_slice: qimage.save(target_img_path)
             elif source_path: shutil.copy2(source_path, target_img_path)

        # Write YOLO format annotation file
        try:
            with open(target_lbl_path, 'w', encoding='utf-8') as f:
                for class_name, class_annotations in annotations.items():
                    if class_name not in class_to_index:
                         print(f"Warning: Class '{class_name}' not in mapping. Skipping annotations in {target_lbl_filename}.")
                         continue
                    class_index = class_to_index[class_name]

                    for ann in class_annotations:
                        line = None
                        if 'segmentation' in ann:
                            seg = ann['segmentation']
                            if isinstance(seg, list) and seg:
                                if isinstance(seg[0], list): seg = seg[0] # Handle nested
                                if len(seg) >= 6:
                                     # Normalize polygon coordinates
                                     normalized_polygon = [
                                         max(0.0, min(1.0, seg[i] / img_width)) if i % 2 == 0
                                         else max(0.0, min(1.0, seg[i] / img_height))
                                         for i in range(len(seg))
                                     ]
                                     line = f"{class_index} " + " ".join(map(lambda x: f"{x:.6f}", normalized_polygon))
                        elif 'bbox' in ann:
                            x, y, w, h = ann['bbox']
                            # Normalize bbox coordinates
                            x_center = max(0.0, min(1.0, (x + w / 2) / img_width))
                            y_center = max(0.0, min(1.0, (y + h / 2) / img_height))
                            norm_w = max(0.0, min(1.0, w / img_width))
                            norm_h = max(0.0, min(1.0, h / img_height))
                            line = f"{class_index} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

                        if line:
                            f.write(line + "\n")
            processed_files += 1
        except Exception as e:
            print(f"Error writing label file {target_lbl_filename}: {e}")
            skipped_files += 1
            # Optionally remove the potentially corrupt label file
            if os.path.exists(target_lbl_path): os.remove(target_lbl_path)


    print(f"YOLO v4 Export: Processed {processed_files} files, Skipped {skipped_files} files.")

    # Create data.yaml file
    names = list(class_mapping.keys())
    yaml_data = {
        'train': os.path.abspath(os.path.join(train_dir, 'images')),
        'val': os.path.abspath(os.path.join(valid_dir, 'images')), # Point to empty valid dir
        'test': '', # Optional: Add a test set path if needed
        'nc': len(names),
        'names': names
    }

    # Save YAML file in the output directory
    yaml_path = os.path.join(output_dir, 'data.yaml')
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f: # Specify encoding
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Saved data.yaml to: {yaml_path}")
    except Exception as e:
        print(f"Error saving data.yaml: {e}")
        # Don't raise here, export might partially succeed

    return os.path.abspath(train_dir), os.path.abspath(yaml_path) # Return absolute paths


def export_yolo_v5plus(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports annotations in YOLO v5+ format."""
    print("Starting YOLO v5+ export...")
    # Create output directories (YOLO v5+ standard structure)
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val') # Create val dir
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')   # Create val dir

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Ensured YOLO v5+ directories exist under: {output_dir}")

    # Create class mapping and slice map (same as v4)
    class_to_index = {name: i for i, name in enumerate(class_mapping.keys())}
    slice_map = {name: img for name, img in slices}

    processed_files = 0
    skipped_files = 0

    # Process all annotations (similar logic to v4)
    for image_name, annotations in all_annotations.items():
        if not annotations or not any(annotations.values()):
            continue

        qimage = None
        source_path = None
        is_slice = False
        target_filename_base = None
        img_width, img_height = 0, 0

        if image_name in slice_map:
             is_slice = True
             qimage = slice_map[image_name]
             target_filename_base = image_name
        elif '_' in image_name and '.' not in image_name:
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice: qimage = matching_slice[1]
             else:
                 for stack_slices in image_slices.values():
                     matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                     if matching_stack_slice: qimage = matching_stack_slice[1]; break
             if qimage: is_slice = True; target_filename_base = image_name
        else:
             source_path = image_paths.get(image_name)
             if source_path and os.path.exists(source_path) and not source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                 target_filename_base = os.path.splitext(image_name)[0]
                 qimage = QImage(source_path)

        if qimage is None or qimage.isNull():
            skipped_files += 1; continue
        img_width, img_height = qimage.width(), qimage.height()
        if img_width <= 0 or img_height <= 0:
            skipped_files += 1; continue

        # Put all in train for simplicity (adjust if split needed)
        target_img_dir = images_train_dir
        target_lbl_dir = labels_train_dir
        image_file_ext = ".png" if is_slice else os.path.splitext(image_name)[1]
        target_img_filename = f"{target_filename_base}{image_file_ext}"
        target_lbl_filename = f"{target_filename_base}.txt"
        target_img_path = os.path.join(target_img_dir, target_img_filename)
        target_lbl_path = os.path.join(target_lbl_dir, target_lbl_filename)

        if not os.path.exists(target_img_path):
             if is_slice: qimage.save(target_img_path)
             elif source_path: shutil.copy2(source_path, target_img_path)

        # Write YOLO label file (same logic as v4)
        try:
            with open(target_lbl_path, 'w', encoding='utf-8') as f:
                for class_name, class_annotations in annotations.items():
                    if class_name not in class_to_index: continue
                    class_index = class_to_index[class_name]
                    for ann in class_annotations:
                        line = None
                        if 'segmentation' in ann:
                            seg = ann['segmentation']
                            if isinstance(seg, list) and seg:
                                if isinstance(seg[0], list): seg = seg[0]
                                if len(seg) >= 6:
                                     norm_poly = [max(0.0, min(1.0, seg[i] / img_width)) if i % 2 == 0 else max(0.0, min(1.0, seg[i] / img_height)) for i in range(len(seg))]
                                     line = f"{class_index} " + " ".join(map(lambda x: f"{x:.6f}", norm_poly))
                        elif 'bbox' in ann:
                            x, y, w, h = ann['bbox']
                            xc = max(0.0, min(1.0, (x + w / 2) / img_width))
                            yc = max(0.0, min(1.0, (y + h / 2) / img_height))
                            nw = max(0.0, min(1.0, w / img_width))
                            nh = max(0.0, min(1.0, h / img_height))
                            line = f"{class_index} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                        if line: f.write(line + "\n")
            processed_files += 1
        except Exception as e:
            print(f"Error writing label file {target_lbl_filename}: {e}")
            skipped_files += 1
            if os.path.exists(target_lbl_path): os.remove(target_lbl_path)

    print(f"YOLO v5+ Export: Processed {processed_files} files, Skipped {skipped_files} files.")

    # Create data.yaml for YOLO v5+
    names = list(class_mapping.keys())
    yaml_data = {
        'path': os.path.abspath(output_dir),  # Root directory (absolute path recommended by YOLO)
        'train': os.path.join('images', 'train'),  # Relative path from 'path'
        'val': os.path.join('images', 'val'),      # Relative path from 'path'
        # 'test': Optional: os.path.join('images', 'test'),
        'nc': len(names),
        'names': names
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f: # Specify encoding
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Saved data.yaml to: {yaml_path}")
    except Exception as e:
        print(f"Error saving data.yaml: {e}")

    return os.path.abspath(output_dir), os.path.abspath(yaml_path) # Return absolute paths


def export_labeled_images(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports images with corresponding instance segmentation masks per class."""
    print("Starting Labeled Images export...")
    # Create output directories
    images_dir = os.path.join(output_dir, 'images') # Original images
    labeled_images_dir = os.path.join(output_dir, 'labeled_images') # Masks
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labeled_images_dir, exist_ok=True)
    print(f"Ensured directories exist: {images_dir}, {labeled_images_dir}")

    # Create a dictionary to store class information for the summary
    class_summary = {class_name: [] for class_name in class_mapping.keys()}
    slice_map = {name: img for name, img in slices}
    processed_files = 0
    skipped_files = 0

    # Create directories for each class inside labeled_images_dir
    for class_name in class_mapping.keys():
        os.makedirs(os.path.join(labeled_images_dir, class_name), exist_ok=True)

    for image_name, annotations in all_annotations.items():
        if not annotations or not any(annotations.values()):
            continue

        # Determine image source and dimensions
        qimage = None
        source_path = None
        is_slice = False
        target_filename_base = None
        img_width, img_height = 0, 0

        if image_name in slice_map:
             is_slice = True; qimage = slice_map[image_name]; target_filename_base = image_name
        elif '_' in image_name and '.' not in image_name:
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice: qimage = matching_slice[1]
             else:
                 for stack_slices in image_slices.values():
                     matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                     if matching_stack_slice: qimage = matching_stack_slice[1]; break
             if qimage: is_slice = True; target_filename_base = image_name
        else:
             source_path = image_paths.get(image_name)
             if source_path and os.path.exists(source_path) and not source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                 target_filename_base = os.path.splitext(image_name)[0]
                 qimage = QImage(source_path)

        if qimage is None or qimage.isNull(): skipped_files += 1; continue
        img_width, img_height = qimage.width(), qimage.height()
        if img_width <= 0 or img_height <= 0: skipped_files += 1; continue

        # Save/Copy original image
        image_file_ext = ".png" if is_slice else os.path.splitext(image_name)[1]
        target_img_filename = f"{target_filename_base}{image_file_ext}"
        target_img_path = os.path.join(images_dir, target_img_filename)
        if not os.path.exists(target_img_path):
             if is_slice: qimage.save(target_img_path)
             elif source_path: shutil.copy2(source_path, target_img_path)

        # Create and save instance masks for each class
        for class_name, class_annotations in annotations.items():
            if not class_annotations: continue # Skip if no annotations for this class

            # Create a mask for this class, uint16 for instance IDs > 255
            instance_mask = np.zeros((img_height, img_width), dtype=np.uint16)
            object_number = 1 # Start instance IDs from 1 for this class mask

            for ann in class_annotations:
                polygon_drawn = False
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, list) and seg:
                        if isinstance(seg[0], list): seg = seg[0]
                        if len(seg) >= 6:
                             try:
                                 polygon = np.array(seg).reshape(-1, 2)
                                 # Ensure coordinates are within bounds before drawing
                                 polygon[:, 0] = np.clip(polygon[:, 0], 0, img_width - 1)
                                 polygon[:, 1] = np.clip(polygon[:, 1], 0, img_height - 1)
                                 # Use skimage.draw.polygon (expects row, col)
                                 rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_height, img_width))
                                 instance_mask[rr, cc] = object_number
                                 polygon_drawn = True
                             except Exception as draw_err:
                                 print(f"Warning: Error drawing polygon for {target_img_filename}, class {class_name}: {draw_err}")

                # Fallback to bbox if segmentation failed or wasn't present
                if not polygon_drawn and 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img_width, x + w), min(img_height, y + h)
                    if x2 > x1 and y2 > y1: # Ensure valid bbox region
                        instance_mask[y1:y2, x1:x2] = object_number

                object_number += 1 # Increment for next instance of this class

            # Save the instance mask if it contains any objects
            if np.any(instance_mask):
                mask_filename = f"{target_filename_base}_{class_name}_inst_mask.png" # Clearer filename
                mask_path = os.path.join(labeled_images_dir, class_name, mask_filename)
                try:
                    # Save as 16-bit PNG
                    Image.fromarray(instance_mask).save(mask_path)
                    # Record file in summary
                    if target_img_filename not in class_summary[class_name]:
                         class_summary[class_name].append(target_img_filename)
                except Exception as save_err:
                    print(f"Error saving mask {mask_path}: {save_err}")

        processed_files += 1

    print(f"Labeled Images Export: Processed {processed_files} files, Skipped {skipped_files} files.")

    # Create summary text file
    summary_path = os.path.join(labeled_images_dir, 'class_summary.txt')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Class Summary (Instance Masks)\n")
            f.write("===============================\n\n")
            f.write("Mask pixel values correspond to instance ID within that class mask file.\n\n")
            for class_name in sorted(class_mapping.keys()):
                files = class_summary.get(class_name)
                if files:  # Only include classes that have annotations
                    f.write(f"Class: {class_name}\n")
                    f.write(f"  Corresponding Original Images: {', '.join(sorted(set(files)))}\n")
                    f.write(f"  Masks in folder: {os.path.join(labeled_images_dir, class_name)}\n\n")
        print(f"Saved class summary to: {summary_path}")
    except Exception as e:
         print(f"Error writing class summary file: {e}")

    return os.path.abspath(labeled_images_dir) # Return path to the masks directory


def export_semantic_labels(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports semantic segmentation masks where pixel value corresponds to class ID."""
    print("Starting Semantic Labels export...")
    # Create output directories
    images_dir = os.path.join(output_dir, 'images') # Original images
    segmented_images_dir = os.path.join(output_dir, 'semantic_masks') # Semantic masks
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(segmented_images_dir, exist_ok=True)
    print(f"Ensured directories exist: {images_dir}, {segmented_images_dir}")

    # Create a mapping of class names to unique pixel values (starting from 1, 0 is background)
    class_to_pixel = {name: i + 1 for i, name in enumerate(sorted(class_mapping.keys()))}
    slice_map = {name: img for name, img in slices}
    processed_files = 0
    skipped_files = 0

    for image_name, annotations in all_annotations.items():
        if not annotations or not any(annotations.values()):
            continue

        # Determine image source and dimensions
        qimage = None
        source_path = None
        is_slice = False
        target_filename_base = None
        img_width, img_height = 0, 0

        if image_name in slice_map:
             is_slice = True; qimage = slice_map[image_name]; target_filename_base = image_name
        elif '_' in image_name and '.' not in image_name:
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice: qimage = matching_slice[1]
             else:
                 for stack_slices in image_slices.values():
                     matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                     if matching_stack_slice: qimage = matching_stack_slice[1]; break
             if qimage: is_slice = True; target_filename_base = image_name
        else:
             source_path = image_paths.get(image_name)
             if source_path and os.path.exists(source_path) and not source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                 target_filename_base = os.path.splitext(image_name)[0]
                 qimage = QImage(source_path)

        if qimage is None or qimage.isNull(): skipped_files += 1; continue
        img_width, img_height = qimage.width(), qimage.height()
        if img_width <= 0 or img_height <= 0: skipped_files += 1; continue

        # Save/Copy original image
        image_file_ext = ".png" if is_slice else os.path.splitext(image_name)[1]
        target_img_filename = f"{target_filename_base}{image_file_ext}"
        target_img_path = os.path.join(images_dir, target_img_filename)
        if not os.path.exists(target_img_path):
             if is_slice: qimage.save(target_img_path)
             elif source_path: shutil.copy2(source_path, target_img_path)

        # Create a single semantic mask (uint8 is usually sufficient for class IDs)
        semantic_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Draw annotations onto the mask, potentially handling overlaps
        # Draw smaller annotations last so they are on top? Or based on class priority?
        # Simple approach: iterate and draw. Last drawn wins in overlapping areas.
        for class_name, class_annotations in annotations.items():
            pixel_value = class_to_pixel.get(class_name)
            if pixel_value is None: continue # Skip if class somehow not in mapping

            for ann in class_annotations:
                polygon_drawn = False
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, list) and seg:
                        if isinstance(seg[0], list): seg = seg[0]
                        if len(seg) >= 6:
                             try:
                                 polygon = np.array(seg).reshape(-1, 2)
                                 polygon[:, 0] = np.clip(polygon[:, 0], 0, img_width - 1)
                                 polygon[:, 1] = np.clip(polygon[:, 1], 0, img_height - 1)
                                 rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_height, img_width))
                                 semantic_mask[rr, cc] = pixel_value
                                 polygon_drawn = True
                             except Exception as draw_err:
                                 print(f"Warning: Error drawing polygon for {target_img_filename}, class {class_name}: {draw_err}")

                if not polygon_drawn and 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img_width, x + w), min(img_height, y + h)
                    if x2 > x1 and y2 > y1:
                        semantic_mask[y1:y2, x1:x2] = pixel_value

        # Save semantic mask as PNG
        mask_filename = f"{target_filename_base}_semantic_mask.png"
        mask_path = os.path.join(segmented_images_dir, mask_filename)
        try:
            Image.fromarray(semantic_mask).save(mask_path)
        except Exception as save_err:
            print(f"Error saving semantic mask {mask_path}: {save_err}")

        processed_files += 1

    print(f"Semantic Labels Export: Processed {processed_files} files, Skipped {skipped_files} files.")

    # Create class mapping text file
    mapping_path = os.path.join(segmented_images_dir, 'class_pixel_mapping.txt')
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("Pixel Value : Class Name\n")
            f.write("------------------------\n")
            f.write("0 : background\n") # Explicitly state background value
            for class_name, pixel_value in sorted(class_to_pixel.items(), key=lambda item: item[1]):
                f.write(f"{pixel_value} : {class_name}\n")
        print(f"Saved class-pixel mapping to: {mapping_path}")
    except Exception as e:
        print(f"Error writing class mapping file: {e}")

    return os.path.abspath(segmented_images_dir) # Return path to semantic masks


def _create_voc_xml(filename, width, height, depth, annotations):
    """Helper function to create the XML structure for Pascal VOC."""
    annotation_node = ET.Element('annotation')

    ET.SubElement(annotation_node, 'folder').text = 'images' # Assuming images are in 'images' subfolder
    ET.SubElement(annotation_node, 'filename').text = filename
    # ET.SubElement(annotation_node, 'path').text = os.path.join('images', filename) # Path optional/relative
    source_node = ET.SubElement(annotation_node, 'source')
    ET.SubElement(source_node, 'database').text = 'Unknown'

    size_node = ET.SubElement(annotation_node, 'size')
    ET.SubElement(size_node, 'width').text = str(width)
    ET.SubElement(size_node, 'height').text = str(height)
    ET.SubElement(size_node, 'depth').text = str(depth)

    ET.SubElement(annotation_node, 'segmented').text = '0' # Default to 0, override if segmentation exists

    has_segmentation = False

    for class_name, class_annotations in annotations.items():
        for ann in class_annotations:
            object_node = ET.SubElement(annotation_node, 'object')
            ET.SubElement(object_node, 'name').text = class_name
            ET.SubElement(object_node, 'pose').text = 'Unspecified'
            ET.SubElement(object_node, 'truncated').text = '0' # Assume not truncated
            ET.SubElement(object_node, 'difficult').text = '0' # Assume not difficult

            bbox = ann.get('bbox')
            segmentation = ann.get('segmentation')

            # Add bounding box if available
            if bbox and len(bbox) == 4:
                x, y, w, h = map(float, bbox)
                xmin = max(0, int(x))
                ymin = max(0, int(y))
                xmax = min(width, int(x + w)) # Clamp to image bounds
                ymax = min(height, int(y + h)) # Clamp to image bounds
                if xmax > xmin and ymax > ymin: # Ensure valid box
                    bndbox_node = ET.SubElement(object_node, 'bndbox')
                    ET.SubElement(bndbox_node, 'xmin').text = str(xmin)
                    ET.SubElement(bndbox_node, 'ymin').text = str(ymin)
                    ET.SubElement(bndbox_node, 'xmax').text = str(xmax)
                    ET.SubElement(bndbox_node, 'ymax').text = str(ymax)

            # Add segmentation if available (for VOC Both format)
            if segmentation:
                if isinstance(segmentation, list) and segmentation:
                    if isinstance(segmentation[0], list): segmentation = segmentation[0]
                    if len(segmentation) >= 6:
                        has_segmentation = True
                        # VOC segmentation format is typically just the list of points
                        segmentation_node = ET.SubElement(object_node, 'segmentation')
                        for i in range(0, len(segmentation), 2):
                            pt_node = ET.SubElement(segmentation_node, 'pt')
                            ET.SubElement(pt_node, 'x').text = str(int(segmentation[i]))
                            ET.SubElement(pt_node, 'y').text = str(int(segmentation[i+1]))

    # Update segmented tag if needed
    if has_segmentation:
        segmented_tag = annotation_node.find('segmented')
        if segmented_tag is not None:
            segmented_tag.text = '1'

    # Pretty print the XML
    xml_string = ET.tostring(annotation_node, encoding='unicode')
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ") # Use 2 spaces for indentation
    return pretty_xml


def _export_pascal_voc_base(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, include_segmentation=False):
    """Base function for Pascal VOC export (BBox only or BBox + Segmentation)."""
    print(f"Starting Pascal VOC export (Include Segmentation: {include_segmentation})...")
    # Create output directories
    images_dir = os.path.join(output_dir, 'JPEGImages') # Standard VOC image folder name
    annotations_dir = os.path.join(output_dir, 'Annotations')
    # segmentation_dir = os.path.join(output_dir, 'SegmentationObject') # Optional for instance masks
    # segmentation_class_dir = os.path.join(output_dir, 'SegmentationClass') # Optional for class masks
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    # if include_segmentation:
    #     os.makedirs(segmentation_dir, exist_ok=True)
    #     os.makedirs(segmentation_class_dir, exist_ok=True)

    print(f"Ensured directories exist: {images_dir}, {annotations_dir}")
    slice_map = {name: img for name, img in slices}
    processed_files = 0
    skipped_files = 0

    for image_name, annotations in all_annotations.items():
        if not annotations or not any(annotations.values()):
            continue

        # Determine image source and dimensions
        qimage = None
        source_path = None
        is_slice = False
        target_filename_base = None
        image_file_ext = None # Keep original extension for VOC filename consistency
        img_width, img_height = 0, 0

        if image_name in slice_map:
             is_slice = True; qimage = slice_map[image_name]; target_filename_base = image_name; image_file_ext=".png"
        elif '_' in image_name and '.' not in image_name:
             matching_slice = next((s for s in slices if s[0] == image_name), None)
             if matching_slice: qimage = matching_slice[1]
             else:
                 for stack_slices in image_slices.values():
                     matching_stack_slice = next((s for s in stack_slices if s[0] == image_name), None)
                     if matching_stack_slice: qimage = matching_stack_slice[1]; break
             if qimage: is_slice = True; target_filename_base = image_name; image_file_ext=".png"
        else:
             source_path = image_paths.get(image_name)
             if source_path and os.path.exists(source_path) and not source_path.lower().endswith(('.tif', '.tiff', '.czi')):
                 target_filename_base = os.path.splitext(image_name)[0]
                 image_file_ext = os.path.splitext(image_name)[1]
                 qimage = QImage(source_path)

        if qimage is None or qimage.isNull(): skipped_files += 1; continue
        img_width, img_height = qimage.width(), qimage.height()
        if img_width <= 0 or img_height <= 0: skipped_files += 1; continue

        # VOC typically uses JPEG, but we'll save/copy originals or PNGs
        # The filename in the XML should match the image file
        target_img_filename = f"{target_filename_base}{image_file_ext}"
        target_img_path = os.path.join(images_dir, target_img_filename)
        target_xml_filename = f"{target_filename_base}.xml"
        target_xml_path = os.path.join(annotations_dir, target_xml_filename)

        # Save/Copy image file
        if not os.path.exists(target_img_path):
             if is_slice: qimage.save(target_img_path)
             elif source_path: shutil.copy2(source_path, target_img_path)

        # Determine image depth (usually 3 for VOC)
        depth = 3 if qimage.format() in [QImage.Format.Format_RGB888, QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32, QImage.Format.Format_ARGB32_Premultiplied] else 1

        # Filter annotations to only include bbox or segmentation based on flag
        filtered_annotations = {}
        for cls, anns in annotations.items():
             filtered_anns_for_cls = []
             for ann in anns:
                  has_bbox = 'bbox' in ann and len(ann['bbox']) == 4
                  has_seg = 'segmentation' in ann and len(ann.get('segmentation',[])) >= 6
                  if has_bbox or (include_segmentation and has_seg):
                       # Ensure bbox exists if segmentation is present
                       if include_segmentation and has_seg and not has_bbox:
                           ann['bbox'] = calculate_bbox(ann['segmentation'])
                       filtered_anns_for_cls.append(ann)
             if filtered_anns_for_cls:
                  filtered_annotations[cls] = filtered_anns_for_cls

        if not filtered_annotations: # Skip if no relevant annotations
            print(f"Skipping {target_img_filename}: No relevant annotations found.")
            continue

        # Create and save XML annotation file
        try:
            xml_content = _create_voc_xml(target_img_filename, img_width, img_height, depth, filtered_annotations)
            with open(target_xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            processed_files += 1
        except Exception as e:
            print(f"Error creating/saving XML for {target_xml_filename}: {e}")
            skipped_files += 1
            # Optionally remove partial XML
            if os.path.exists(target_xml_path): os.remove(target_xml_path)


    print(f"Pascal VOC Export: Processed {processed_files} files, Skipped {skipped_files} files.")
    return os.path.abspath(output_dir) # Return the main output directory path


def export_pascal_voc_bbox(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports annotations in Pascal VOC format (Bounding Boxes only)."""
    return _export_pascal_voc_base(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, include_segmentation=False)

def export_pascal_voc_both(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Exports annotations in Pascal VOC format (Bounding Boxes and Segmentation)."""
    return _export_pascal_voc_base(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, include_segmentation=True)
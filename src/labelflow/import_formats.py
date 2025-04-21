# import_formats.py (Migrated to PyQt6)
import json
import os
import yaml
from PIL import Image
import traceback # For detailed error logging

# PyQt6 Imports (Potentially needed if UI elements were added here, currently none)
# from PyQt6.QtCore import QRectF
# from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox # Still needed for messages

def import_coco_json(file_path, class_mapping):
    """Imports annotations from a COCO JSON file."""
    print(f"Importing COCO JSON from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            coco_data = json.load(f)

        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(f"Missing required field '{field}' in JSON file: {file_path}")

        imported_annotations = {} # image_filename -> {category_name: [annotations]}
        image_info = {}           # image_id -> {image_details}

        # Create reverse mapping of category IDs to names
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        if not category_id_to_name:
             print("Warning: No categories found in COCO JSON.")

        # Determine the image directory (usually relative to the JSON file)
        json_dir = os.path.dirname(file_path)
        # Common convention is an 'images' subdirectory, but could be different
        # Look for image paths *within* the image dictionaries first
        images_root_dir = json_dir # Default search start

        # Process images - store path relative to JSON if possible, else absolute
        print(f"Processing {len(coco_data.get('images',[]))} image entries...")
        for image in coco_data.get('images', []):
            try:
                img_id = int(image['id'])
                file_name = image['file_name']
                width = int(image['width'])
                height = int(image['height'])

                # Try to construct path relative to JSON first
                potential_path = os.path.join(json_dir, file_name)
                if not os.path.exists(potential_path):
                     # Try 'images' subdirectory
                     potential_path = os.path.join(json_dir, 'images', file_name)
                     if not os.path.exists(potential_path):
                          # Try absolute path if stored in JSON (less common)
                          potential_path = image.get('coco_url', '') or image.get('path', '') # Check common keys
                          if not os.path.isabs(potential_path): potential_path = "" # Ignore relative paths we can't resolve

                # Use the first valid path found, or store None if not found
                image_path = potential_path if os.path.exists(potential_path) else None

                image_info[img_id] = {
                    'file_name': file_name,
                    'width': width,
                    'height': height,
                    'path': image_path, # Store the resolved path or None
                    'id': img_id
                }
                # if image_path is None:
                #      print(f"Warning: Image file not found for entry: {file_name}")

            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid image entry: {image}. Error: {e}")
                continue

        # Process annotations
        print(f"Processing {len(coco_data.get('annotations',[]))} annotation entries...")
        annotations_processed = 0
        annotations_skipped = 0
        for ann in coco_data.get('annotations', []):
            try:
                image_id = int(ann['image_id'])
                category_id = int(ann['category_id'])

                # Check if annotation references a valid image and category
                if image_id not in image_info:
                    # print(f"Warning: Annotation refers to non-existent image ID: {image_id}. Skipping.")
                    annotations_skipped += 1
                    continue
                if category_id not in category_id_to_name:
                    # print(f"Warning: Annotation refers to non-existent category ID: {category_id}. Skipping.")
                    annotations_skipped += 1
                    continue

                file_name = image_info[image_id]['file_name']
                category_name = category_id_to_name[category_id]

                if file_name not in imported_annotations:
                    imported_annotations[file_name] = {}
                if category_name not in imported_annotations[file_name]:
                    imported_annotations[file_name][category_name] = []

                # Prepare the annotation dictionary for internal use
                internal_annotation = {
                    'category_id': category_id,
                    'category_name': category_name,
                    'area': float(ann.get('area', 0.0)),
                    # Store source ID if available, useful for debugging
                    'source_id': ann.get('id')
                }

                segmentation = ann.get('segmentation')
                bbox = ann.get('bbox')
                annotation_type = None

                # Handle segmentation data
                if segmentation and isinstance(segmentation, list) and segmentation:
                    # COCO can have [[poly1], [poly2]] or RLE, we handle simple polygon lists here
                    if isinstance(segmentation[0], list): # Take first polygon if multiple
                        poly_coords = segmentation[0]
                    else: # Assume single flat list polygon
                        poly_coords = segmentation

                    if len(poly_coords) >= 6: # Need at least 3 points
                        internal_annotation['segmentation'] = [float(p) for p in poly_coords]
                        annotation_type = 'polygon'
                    #else: print(f"Warning: Skipping invalid polygon (less than 3 points) for ann ID {ann.get('id')}")

                # Handle bbox data
                if bbox and len(bbox) == 4:
                    internal_annotation['bbox'] = [float(b) for b in bbox] # Store as x,y,w,h
                    if annotation_type is None: # Use bbox as primary if no valid segmentation
                        annotation_type = 'rectangle' # Internal type name
                # else: print(f"Warning: Skipping invalid bbox for ann ID {ann.get('id')}")

                # Assign type and add to list if valid
                if annotation_type:
                    internal_annotation['type'] = annotation_type
                    imported_annotations[file_name][category_name].append(internal_annotation)
                    annotations_processed += 1
                else:
                    # print(f"Warning: Skipping annotation ID {ann.get('id')} due to missing/invalid segmentation or bbox.")
                    annotations_skipped += 1

            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Error processing annotation: {ann}. Error: {e}")
                annotations_skipped += 1
                continue

        print(f"COCO Import Summary: Processed={annotations_processed}, Skipped={annotations_skipped}")
        return imported_annotations, image_info

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {file_path}. Error: {e}")
    except Exception as e:
        print(f"Unexpected error during COCO import: {e}")
        traceback.print_exc()
        raise ValueError(f"Error importing COCO JSON: {e}")


def import_yolo_v4(yaml_file_path, class_mapping):
    """Imports annotations from YOLO v4 format."""
    print(f"Importing YOLO v4 from YAML: {yaml_file_path}")
    if not os.path.exists(yaml_file_path):
        raise ValueError(f"YAML file not found: {yaml_file_path}")

    directory_path = os.path.dirname(yaml_file_path)

    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f: # Specify encoding
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error reading YAML file {yaml_file_path}: {e}")

    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names ('names') found in the YAML file.")

    # YOLO v4 expects 'train' and 'val' paths relative to YAML usually
    train_img_path_rel = yaml_data.get('train')
    val_img_path_rel = yaml_data.get('val')

    if not train_img_path_rel:
        raise ValueError("YAML file must contain a 'train' path (relative path to training images).")

    # Construct absolute paths based on YAML location
    train_images_dir = os.path.abspath(os.path.join(directory_path, train_img_path_rel))
    # Labels are typically in a parallel 'labels' directory
    train_labels_dir = os.path.join(os.path.dirname(train_images_dir), 'labels')

    if not os.path.isdir(train_images_dir):
        raise ValueError(f"Training images directory not found: {train_images_dir}")
    if not os.path.isdir(train_labels_dir):
        raise ValueError(f"Training labels directory not found: {train_labels_dir}")

    imported_annotations = {}
    image_info = {}
    processed_labels = 0
    skipped_annotations = 0
    missing_images_count = 0

    print(f"Searching for labels in: {train_labels_dir}")
    for label_file in os.listdir(train_labels_dir):
        if label_file.lower().endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            img_file = None
            img_path = None

            # Find corresponding image file (case-insensitive check common extensions)
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                potential_img_file = base_name + ext
                potential_img_path = os.path.join(train_images_dir, potential_img_file)
                if os.path.exists(potential_img_path):
                    img_file = potential_img_file
                    img_path = potential_img_path
                    break

            if img_path is None:
                # print(f"Warning: No corresponding image found for label: {label_file}")
                missing_images_count += 1
                continue

            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                 print(f"Warning: Could not read image dimensions for {img_file}: {e}. Skipping.")
                 missing_images_count += 1
                 continue

            image_id = len(image_info) + 1
            image_info[image_id] = {
                'file_name': img_file,
                'width': img_width,
                'height': img_height,
                'id': image_id,
                'path': img_path
            }

            # Ensure entry exists even if label file is empty
            if img_file not in imported_annotations:
                 imported_annotations[img_file] = {}

            label_path = os.path.join(train_labels_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as f: # Specify encoding
                    lines = f.readlines()

                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts: continue # Skip empty lines

                    try:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]

                        if class_id < 0 or class_id >= len(class_names):
                            print(f"Warning: Invalid class ID {class_id} in {label_file}, line {line_num+1}. Skipping.")
                            skipped_annotations += 1
                            continue
                        class_name = class_names[class_id]

                        if class_name not in imported_annotations[img_file]:
                            imported_annotations[img_file][class_name] = []

                        annotation = {
                            'category_id': class_id, # Use the index from YAML as ID
                            'category_name': class_name,
                            'source_line': line_num + 1
                        }

                        if len(coords) == 4:  # Bounding box format (xc, yc, w, h)
                            xc, yc, w, h = coords
                            # Convert YOLO format (normalized) to internal bbox (absolute x,y,w,h)
                            abs_w = w * img_width
                            abs_h = h * img_height
                            abs_x = (xc * img_width) - (abs_w / 2)
                            abs_y = (yc * img_height) - (abs_h / 2)
                            annotation['bbox'] = [abs_x, abs_y, abs_w, abs_h]
                            annotation['type'] = 'rectangle' # Or 'bbox' if preferred internally
                        elif len(coords) >= 6 and len(coords) % 2 == 0: # Polygon format (xn1, yn1, xn2, yn2...)
                            # Convert normalized polygon to absolute coordinates
                            abs_polygon = []
                            for i in range(0, len(coords), 2):
                                abs_x = coords[i] * img_width
                                abs_y = coords[i+1] * img_height
                                abs_polygon.extend([abs_x, abs_y])
                            annotation['segmentation'] = abs_polygon
                            annotation['type'] = 'polygon'
                        else:
                            print(f"Warning: Invalid coordinate count ({len(coords)}) in {label_file}, line {line_num+1}. Skipping.")
                            skipped_annotations += 1
                            continue

                        imported_annotations[img_file][class_name].append(annotation)

                    except (ValueError, IndexError) as parse_err:
                        print(f"Warning: Error parsing line {line_num+1} in {label_file}: {parse_err}. Skipping line.")
                        skipped_annotations += 1
                        continue
                processed_labels += 1
            except Exception as file_err:
                 print(f"Warning: Error reading label file {label_file}: {file_err}")
                 skipped_annotations += len(lines) # Assume all annotations skipped for this file

    print(f"YOLO v4 Import Summary: Processed={processed_labels} label files, Skipped Annotations={skipped_annotations}, Missing Images={missing_images_count}")

    if missing_images_count > 0:
         QMessageBox.warning(None, "Import Warning", f"Could not find corresponding images for {missing_images_count} label files. These labels were skipped.")

    return imported_annotations, image_info


def import_yolo_v5plus(yaml_file_path, class_mapping):
    """Imports annotations from YOLO v5+ format."""
    print(f"Importing YOLO v5+ from YAML: {yaml_file_path}")
    if not os.path.exists(yaml_file_path):
        raise ValueError(f"YAML file not found: {yaml_file_path}")

    root_dir = os.path.dirname(yaml_file_path)

    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f: # Specify encoding
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error reading YAML file {yaml_file_path}: {e}")

    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names ('names') found in the YAML file.")

    imported_annotations = {}
    image_info = {}
    processed_labels = 0
    skipped_annotations = 0
    missing_images_count = 0

    # Process both train and val splits if defined
    for split in ['train', 'val']:
        img_path_rel = yaml_data.get(split)
        if not img_path_rel:
            print(f"Info: '{split}' path not defined in YAML, skipping this split.")
            continue

        # Construct absolute paths (YOLOv5+ often uses paths relative to YAML)
        images_dir = os.path.abspath(os.path.join(root_dir, img_path_rel))
        # Labels are expected parallel to images (e.g., ../labels/train)
        labels_dir = os.path.abspath(os.path.join(os.path.dirname(images_dir), '..', 'labels', os.path.basename(images_dir)))

        print(f"Processing split '{split}':")
        print(f"  Images dir: {images_dir}")
        print(f"  Labels dir: {labels_dir}")

        if not os.path.isdir(images_dir):
            print(f"Warning: Images directory not found for split '{split}': {images_dir}. Skipping.")
            continue
        if not os.path.isdir(labels_dir):
            print(f"Warning: Labels directory not found for split '{split}': {labels_dir}. Skipping.")
            continue

        for label_file in os.listdir(labels_dir):
            if label_file.lower().endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                img_file = None
                img_path = None

                # Find corresponding image file
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    potential_img_file = base_name + ext
                    potential_img_path = os.path.join(images_dir, potential_img_file)
                    if os.path.exists(potential_img_path):
                        img_file = potential_img_file
                        img_path = potential_img_path
                        break

                if img_path is None:
                    missing_images_count += 1
                    continue

                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                     print(f"Warning: Could not read image dimensions for {img_file}: {e}. Skipping.")
                     missing_images_count += 1
                     continue

                # Use existing image ID if file already processed (e.g., present in both train/val)
                existing_entry = next((item for item in image_info.values() if item['file_name'] == img_file), None)
                if existing_entry:
                     image_id = existing_entry['id']
                else:
                     image_id = len(image_info) + 1
                     image_info[image_id] = {
                         'file_name': img_file, 'width': img_width, 'height': img_height,
                         'id': image_id, 'path': img_path
                     }

                # Ensure entry exists even if label file is empty
                if img_file not in imported_annotations:
                     imported_annotations[img_file] = {}

                label_path = os.path.join(labels_dir, label_file)
                try:
                    with open(label_path, 'r', encoding='utf-8') as f: # Specify encoding
                        lines = f.readlines()

                    for line_num, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts: continue

                        try:
                            class_id = int(parts[0])
                            coords = [float(p) for p in parts[1:]]

                            if class_id < 0 or class_id >= len(class_names):
                                print(f"Warning: Invalid class ID {class_id} in {label_file}, line {line_num+1}. Skipping.")
                                skipped_annotations += 1
                                continue
                            class_name = class_names[class_id]

                            if class_name not in imported_annotations[img_file]:
                                imported_annotations[img_file][class_name] = []

                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'source_line': line_num + 1,
                                'source_split': split # Record which split it came from
                            }

                            if len(coords) == 4: # Bounding box
                                xc, yc, w, h = coords
                                abs_w = w * img_width
                                abs_h = h * img_height
                                abs_x = (xc * img_width) - (abs_w / 2)
                                abs_y = (yc * img_height) - (abs_h / 2)
                                annotation['bbox'] = [abs_x, abs_y, abs_w, abs_h]
                                annotation['type'] = 'rectangle'
                            elif len(coords) >= 6 and len(coords) % 2 == 0: # Polygon
                                abs_polygon = []
                                for i in range(0, len(coords), 2):
                                    abs_x = coords[i] * img_width
                                    abs_y = coords[i+1] * img_height
                                    abs_polygon.extend([abs_x, abs_y])
                                annotation['segmentation'] = abs_polygon
                                annotation['type'] = 'polygon'
                            else:
                                print(f"Warning: Invalid coordinate count ({len(coords)}) in {label_file}, line {line_num+1}. Skipping.")
                                skipped_annotations += 1
                                continue

                            imported_annotations[img_file][class_name].append(annotation)

                        except (ValueError, IndexError) as parse_err:
                            print(f"Warning: Error parsing line {line_num+1} in {label_file}: {parse_err}. Skipping line.")
                            skipped_annotations += 1
                            continue
                    processed_labels += 1
                except Exception as file_err:
                    print(f"Warning: Error reading label file {label_file}: {file_err}")
                    skipped_annotations += len(lines) # Estimate

    print(f"YOLO v5+ Import Summary: Processed={processed_labels} label files, Skipped Annotations={skipped_annotations}, Missing Images={missing_images_count}")

    if missing_images_count > 0:
         QMessageBox.warning(None, "Import Warning", f"Could not find corresponding images for {missing_images_count} label files. These labels were skipped.")

    return imported_annotations, image_info


def process_import_format(import_format, file_path, class_mapping):
    """Calls the appropriate import function based on the selected format."""
    try:
        if import_format == "COCO JSON":
            return import_coco_json(file_path, class_mapping)
        elif import_format == "YOLO (v4 and earlier)":
            return import_yolo_v4(file_path, class_mapping)
        elif import_format == "YOLO (v5+)":
            return import_yolo_v5plus(file_path, class_mapping)
        else:
            raise ValueError(f"Unsupported import format: {import_format}")
    except Exception as e:
        # Catch potential errors from import functions and raise consistently
        print(f"Error during import ({import_format}): {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to import from {import_format}: {e}")
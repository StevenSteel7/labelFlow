# dataset_splitter.py (Migrated to PyQt6)
import os
import json
import shutil
import random
# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                             QLabel, QSpinBox, QRadioButton, QButtonGroup, QMessageBox, QComboBox,
                             QApplication) # Added QApplication for processEvents
from PyQt6.QtCore import Qt
import yaml
from PIL import Image

class DatasetSplitterTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Splitter")
        self.setGeometry(100, 100, 500, 300)
        # Use Qt.WindowType enum for flags
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.input_directory = "" # Initialize attributes
        self.output_directory = ""
        self.json_file = ""
        self.coco_data = None # Initialize coco_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Option selection
        options_layout = QVBoxLayout()
        self.images_only_radio = QRadioButton("Images Only")
        options_layout.addWidget(self.images_only_radio)

        images_annotations_layout = QHBoxLayout()
        self.images_annotations_radio = QRadioButton("Images and Annotations")
        images_annotations_layout.addWidget(self.images_annotations_radio)
        self.select_json_button = QPushButton("Upload COCO JSON File")
        self.select_json_button.clicked.connect(self.select_json_file)
        self.select_json_button.setEnabled(False)
        images_annotations_layout.addWidget(self.select_json_button)
        options_layout.addLayout(images_annotations_layout)

        # Format selection (Only enabled when Annotations are selected)
        self.format_selection_layout = QHBoxLayout()
        self.format_label = QLabel("Output Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["COCO JSON", "YOLO"])
        self.format_combo.setEnabled(False) # Disabled by default
        self.format_selection_layout.addWidget(self.format_label)
        self.format_selection_layout.addWidget(self.format_combo)
        options_layout.addLayout(self.format_selection_layout)

        layout.addLayout(options_layout)

        option_group = QButtonGroup(self)
        option_group.addButton(self.images_only_radio)
        option_group.addButton(self.images_annotations_radio)

        self.images_only_radio.setChecked(True) # Default selection

        # Percentage inputs
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train %:"))
        self.train_percent = QSpinBox()
        self.train_percent.setRange(0, 100)
        self.train_percent.setValue(70)
        train_layout.addWidget(self.train_percent)
        layout.addLayout(train_layout)

        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation %:"))
        self.val_percent = QSpinBox()
        self.val_percent.setRange(0, 100)
        self.val_percent.setValue(30)
        val_layout.addWidget(self.val_percent)
        layout.addLayout(val_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test %:"))
        self.test_percent = QSpinBox()
        self.test_percent.setRange(0, 100)
        self.test_percent.setValue(0)
        test_layout.addWidget(self.test_percent)
        layout.addLayout(test_layout)


        # Buttons
        self.select_input_button = QPushButton("Select Input Directory")
        self.select_input_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.select_input_button)

        self.select_output_button = QPushButton("Select Output Directory")
        self.select_output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.select_output_button)

        self.split_button = QPushButton("Split Dataset")
        self.split_button.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_button)

        self.setLayout(layout)


        # Connect radio buttons to enable/disable JSON selection and format combo
        self.images_only_radio.toggled.connect(self.toggle_annotation_options)
        self.images_annotations_radio.toggled.connect(self.toggle_annotation_options)

    def toggle_annotation_options(self):
        is_annotations = self.images_annotations_radio.isChecked()
        self.select_json_button.setEnabled(is_annotations)
        self.format_combo.setEnabled(is_annotations)
        # If switching back to Images Only, clear JSON selection
        if not is_annotations:
            self.json_file = ""
            self.coco_data = None

    def select_input_directory(self):
        # QFileDialog.getExistingDirectory returns a string path
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_directory = directory
            print(f"Input directory selected: {self.input_directory}") # Debug print
        else:
            print("Input directory selection cancelled.")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_directory = directory
            print(f"Output directory selected: {self.output_directory}") # Debug print
        else:
             print("Output directory selection cancelled.")

    def select_json_file(self):
        # QFileDialog.getOpenFileName returns tuple (fileName, selectedFilter)
        file_name, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON File", "", "JSON Files (*.json)")
        if file_name:
            self.json_file = file_name
            try:
                 with open(self.json_file, 'r', encoding='utf-8') as f: # Add encoding
                     self.coco_data = json.load(f)
                 print(f"COCO JSON file selected and loaded: {self.json_file}") # Debug print
            except Exception as e:
                 QMessageBox.critical(self, "Error Loading JSON", f"Failed to load or parse JSON file:\n{e}")
                 self.json_file = "" # Reset if loading failed
                 self.coco_data = None
        else:
             print("COCO JSON selection cancelled.")

    def split_dataset(self):
        if not self.input_directory or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select input and output directories.")
            return

        if self.images_annotations_radio.isChecked() and (not self.json_file or not self.coco_data):
            QMessageBox.warning(self, "Error", "Please select a valid COCO JSON file when splitting with annotations.")
            return

        train_percent = self.train_percent.value()
        val_percent = self.val_percent.value()
        test_percent = self.test_percent.value()

        if train_percent + val_percent + test_percent != 100:
            QMessageBox.warning(self, "Error", "Percentages must add up to 100%.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Use enum
        try:
            if self.images_only_radio.isChecked():
                self.split_images_only(train_percent, val_percent, test_percent)
            else:
                # Pass loaded coco_data to avoid reloading
                self.split_images_and_annotations(self.coco_data, train_percent, val_percent, test_percent)
            QMessageBox.information(self, "Success", "Dataset split successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Split Error", f"An error occurred during splitting:\n{e}")
            import traceback
            traceback.print_exc() # Print detailed traceback to console
        finally:
             QApplication.restoreOverrideCursor()


    def split_images_only(self, train_p, val_p, test_p):
        print("Splitting images only...")
        try:
            image_files = [f for f in os.listdir(self.input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            if not image_files:
                 raise ValueError("No image files found in the input directory.")
            print(f"Found {len(image_files)} images.")

            random.shuffle(image_files)
            n_total = len(image_files)
            n_train = int(n_total * train_p / 100)
            n_val = int(n_total * val_p / 100)
            # n_test is the remainder

            train_images = image_files[:n_train]
            val_images = image_files[n_train : n_train + n_val]
            test_images = image_files[n_train + n_val :]

            print(f"Splits: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

            for subset, images in [("train", train_images),
                                 ("val", val_images),
                                 ("test", test_images)]:
                if images:  # Only create directories and copy images if there are images for this split
                    subset_dir = os.path.join(self.output_directory, subset)
                    os.makedirs(subset_dir, exist_ok=True)
                    print(f"Copying {len(images)} images to {subset}...")
                    self.copy_images(images, subset, images_only=True)

        except Exception as e:
            print(f"Error in split_images_only: {e}")
            raise # Re-raise the exception to be caught by the main handler

    def split_images_and_annotations(self, coco_data, train_p, val_p, test_p):
        print("Splitting images and annotations...")
        if 'images' not in coco_data or not coco_data['images']:
             raise ValueError("COCO JSON data does not contain 'images' information or it's empty.")

        all_image_info = coco_data['images'] # List of image dictionaries
        random.shuffle(all_image_info)

        n_total = len(all_image_info)
        n_train = int(n_total * train_p / 100)
        n_val = int(n_total * val_p / 100)

        train_image_info = all_image_info[:n_train]
        val_image_info = all_image_info[n_train : n_train + n_val]
        test_image_info = all_image_info[n_train + n_val :]

        print(f"Splits: Train={len(train_image_info)}, Val={len(val_image_info)}, Test={len(test_image_info)}")

        # Create main directories
        os.makedirs(self.output_directory, exist_ok=True)

        output_format = self.format_combo.currentText()
        print(f"Selected output format: {output_format}")

        if output_format == "COCO JSON":
            self.split_coco_format(coco_data, train_image_info, val_image_info, test_image_info)
        elif output_format == "YOLO":
            self.split_yolo_format(coco_data, train_image_info, val_image_info, test_image_info)
        else:
             raise ValueError(f"Unsupported output format: {output_format}")

    def copy_images(self, image_filenames, subset, images_only=False):
        if not image_filenames:
            return

        if images_only:
            subset_dir = os.path.join(self.output_directory, subset)
        else:
            # Standard structure: output_dir/subset/images
            subset_dir = os.path.join(self.output_directory, subset, "images")

        os.makedirs(subset_dir, exist_ok=True)

        copied_count = 0
        skipped_count = 0
        for image_filename in image_filenames:
            src = os.path.join(self.input_directory, image_filename)
            dst = os.path.join(subset_dir, image_filename)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Failed to copy {src} to {dst}: {e}")
                    skipped_count += 1
            else:
                 print(f"Warning: Source image not found, skipping copy: {src}")
                 skipped_count += 1
        print(f"Copied {copied_count} images, skipped {skipped_count} for subset {subset}.")


    def create_subset_coco_annotations(self, coco_data, subset_image_info):
        subset_image_ids = {img['id'] for img in subset_image_info} # Use a set for faster lookup

        # Filter annotations based on the image IDs in the subset
        subset_annotations = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] in subset_image_ids]

        # Return the new COCO structure for this subset
        return {
            "images": subset_image_info, # Already filtered list of image dicts
            "annotations": subset_annotations,
            "categories": coco_data.get('categories', []) # Categories remain the same
        }

    def split_coco_format(self, coco_data, train_image_info, val_image_info, test_image_info):
        print("Splitting in COCO format...")
        # Only create directories and save annotations for non-empty splits
        for subset_name, subset_info in [("train", train_image_info),
                                         ("val", val_image_info),
                                         ("test", test_image_info)]:
            if subset_info:  # Only process if there are images in this split
                print(f"Processing subset: {subset_name}")
                subset_dir = os.path.join(self.output_directory, subset_name)
                images_dir = os.path.join(subset_dir, "images") # COCO expects images in a subdir usually
                os.makedirs(images_dir, exist_ok=True)

                # Get just the filenames for copying
                image_filenames = [img['file_name'] for img in subset_info]
                self.copy_images(image_filenames, subset_name, images_only=False) # Copy to subset/images

                # Create and save annotations file for this subset
                subset_data = self.create_subset_coco_annotations(coco_data, subset_info)
                self.save_coco_annotations(subset_data, subset_name)
            else:
                print(f"Skipping empty subset: {subset_name}")


    def save_coco_annotations(self, data, subset_name):
        # Annotations file goes in the subset directory (e.g., output/train/annotations.json)
        subset_dir = os.path.join(self.output_directory, subset_name)
        os.makedirs(subset_dir, exist_ok=True) # Ensure subset dir exists
        output_file = os.path.join(subset_dir, "annotations.json") # Standard name often used
        try:
            with open(output_file, 'w', encoding='utf-8') as f: # Specify encoding
                json.dump(data, f, indent=2)
            print(f"Saved COCO annotations for {subset_name} to {output_file}")
        except Exception as e:
             print(f"Error saving COCO annotations for {subset_name}: {e}")
             raise

    def split_yolo_format(self, coco_data, train_image_info, val_image_info, test_image_info):
        print("Splitting in YOLO format...")
        # Create class mapping (COCO category ID to YOLO index 0..N-1)
        if 'categories' not in coco_data:
             raise ValueError("COCO data missing 'categories' information.")
        categories = {cat["id"]: i for i, cat in enumerate(coco_data["categories"])}
        yolo_class_names = [cat["name"] for cat in sorted(coco_data["categories"], key=lambda x: categories[x["id"]])]

        yaml_paths = {} # Store relative paths for data.yaml

        # Process each non-empty subset
        for subset_name, subset_info in [("train", train_image_info),
                                         ("val", val_image_info),
                                         ("test", test_image_info)]:
            if not subset_info:  # Skip if no images in this split
                print(f"Skipping empty subset: {subset_name}")
                continue

            print(f"Processing subset: {subset_name}")
            # Define YOLO standard directory structure relative to output_dir
            images_dir_rel = os.path.join(subset_name, "images")
            labels_dir_rel = os.path.join(subset_name, "labels")
            images_dir_abs = os.path.join(self.output_directory, images_dir_rel)
            labels_dir_abs = os.path.join(self.output_directory, labels_dir_rel)
            os.makedirs(images_dir_abs, exist_ok=True)
            os.makedirs(labels_dir_abs, exist_ok=True)
            yaml_paths[subset_name] = images_dir_rel # Store relative path for yaml

            # Get annotations relevant to this subset
            subset_image_ids = {img['id'] for img in subset_info}
            subset_annotations = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] in subset_image_ids]

            # Group annotations by image ID for easier processing
            annotations_by_image = {}
            for ann in subset_annotations:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)

            # Process each image in the subset
            for image_data in subset_info:
                image_file = image_data['file_name']
                image_id = image_data['id']
                img_width = image_data['width']
                img_height = image_data['height']

                # Copy image
                src = os.path.join(self.input_directory, image_file)
                if os.path.exists(src):
                     shutil.copy2(src, os.path.join(images_dir_abs, image_file))
                else:
                     print(f"Warning: Source image not found, skipping copy: {src}")
                     continue # Skip if source image doesn't exist

                # Create YOLO format labels if annotations exist for this image
                label_file = os.path.join(labels_dir_abs, os.path.splitext(image_file)[0] + ".txt")
                with open(label_file, "w", encoding='utf-8') as f: # Specify encoding
                    if image_id in annotations_by_image:
                        for ann in annotations_by_image[image_id]:
                            # Convert COCO class id to YOLO class id
                            yolo_class = categories.get(ann["category_id"])
                            if yolo_class is None:
                                 print(f"Warning: Category ID {ann['category_id']} not found in mapping. Skipping annotation.")
                                 continue

                            # Convert COCO bbox [x,y,w,h] to YOLO format [x_center, y_center, width, height] (normalized)
                            if 'bbox' in ann:
                                x, y, w, h = ann["bbox"]
                                x_center = (x + w / 2) / img_width
                                y_center = (y + h / 2) / img_height
                                norm_w = w / img_width
                                norm_h = h / img_height

                                # Clamp values to [0, 1]
                                x_center = max(0.0, min(1.0, x_center))
                                y_center = max(0.0, min(1.0, y_center))
                                norm_w = max(0.0, min(1.0, norm_w))
                                norm_h = max(0.0, min(1.0, norm_h))

                                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                            # Add support for segmentation if needed (YOLOv5+ format)
                            elif 'segmentation' in ann and ann['segmentation']:
                                seg = ann['segmentation']
                                if isinstance(seg[0], list): seg = seg[0] # Take first polygon if nested
                                if len(seg) >= 6:
                                     normalized_poly = []
                                     for i in range(0, len(seg), 2):
                                          px = max(0.0, min(1.0, seg[i] / img_width))
                                          py = max(0.0, min(1.0, seg[i+1] / img_height))
                                          normalized_poly.extend([f"{px:.6f}", f"{py:.6f}"])
                                     f.write(f"{yolo_class} {' '.join(normalized_poly)}\n")

        # Create data.yaml with relative paths
        yaml_data = {
            'path': os.path.abspath(self.output_directory), # Absolute path to dataset root
            'train': yaml_paths.get('train', ''), # Relative path from root
            'val': yaml_paths.get('val', ''),     # Relative path from root
            'test': yaml_paths.get('test', ''),   # Relative path from root
            'nc': len(yolo_class_names),
            'names': yolo_class_names
        }

        try:
            with open(os.path.join(self.output_directory, 'data.yaml'), 'w', encoding='utf-8') as f: # Specify encoding
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            print(f"Saved data.yaml to {os.path.join(self.output_directory, 'data.yaml')}")
        except Exception as e:
             print(f"Error saving data.yaml: {e}")
             raise


    def show_centered(self, parent):
        # Geometry logic remains the same
        if parent:
            parent_geo = parent.geometry()
            self.move(parent_geo.center() - self.rect().center())
        self.show()
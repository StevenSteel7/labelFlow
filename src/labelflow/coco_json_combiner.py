# coco_json_combiner.py (Migrated to PyQt6)

import json
import os
# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QMessageBox, QApplication)
from PyQt6.QtCore import Qt

class COCOJSONCombinerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("COCO JSON Combiner")
        self.setGeometry(100, 100, 400, 300)
        # Use PyQt6 enums for flags and modality
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.json_files = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.file_labels = []
        for i in range(5):
            file_layout = QHBoxLayout()
            label = QLabel(f"File {i+1}: Not selected")
            self.file_labels.append(label)
            file_layout.addWidget(label)
            select_button = QPushButton(f"Select File {i+1}")
            # Use lambda with default argument capture for index 'x'
            select_button.clicked.connect(lambda checked=False, x=i: self.select_file(x))
            file_layout.addWidget(select_button)
            layout.addLayout(file_layout)

        self.combine_button = QPushButton("Combine JSON Files")
        self.combine_button.clicked.connect(self.combine_json_files)
        self.combine_button.setEnabled(False)
        layout.addWidget(self.combine_button)

        self.setLayout(layout)

    def select_file(self, index):
        # QFileDialog.getOpenFileName returns tuple (fileName, selectedFilter)
        file_name, _ = QFileDialog.getOpenFileName(self, f"Select COCO JSON File {index+1}", "", "JSON Files (*.json)")
        if file_name:
            if file_name not in self.json_files:
                self.json_files.append(file_name)
                self.file_labels[index].setText(f"File {index+1}: {os.path.basename(file_name)}")
                self.combine_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Duplicate File", "This file has already been selected.")
        QApplication.processEvents() # Process events to update UI immediately


    def combine_json_files(self):
        if not self.json_files:
            QMessageBox.warning(self, "No Files", "Please select at least one JSON file to combine.")
            return

        combined_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        image_file_names = set()
        category_name_to_new_id = {} # Map category name to its ID in the combined file
        next_image_id = 1
        next_annotation_id = 1
        next_category_id = 1 # Start category IDs from 1

        try:
            for file_path in self.json_files:
                with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
                    data = json.load(f)

                current_file_cat_id_map = {} # Map old category ID in this file to new combined ID

                # Combine categories ensuring unique names
                for category in data.get('categories', []):
                    cat_name = category['name']
                    if cat_name not in category_name_to_new_id:
                        category_name_to_new_id[cat_name] = next_category_id
                        current_file_cat_id_map[category['id']] = next_category_id
                        # Add new category to combined data
                        new_cat = category.copy() # Avoid modifying original category dict directly
                        new_cat['id'] = next_category_id
                        combined_data['categories'].append(new_cat)
                        next_category_id += 1
                    else:
                        # Category name already exists, map old ID to existing new ID
                        current_file_cat_id_map[category['id']] = category_name_to_new_id[cat_name]


                # Combine images and annotations
                image_id_map = {} # Map old image ID in this file to new combined ID
                for image in data.get('images', []):
                    # Skip duplicate images based on file_name
                    if image['file_name'] not in image_file_names:
                        image_file_names.add(image['file_name'])
                        image_id_map[image['id']] = next_image_id
                        new_image = image.copy() # Avoid modifying original
                        new_image['id'] = next_image_id
                        combined_data['images'].append(new_image)
                        next_image_id += 1
                    # If image is duplicate, find its existing new ID for annotation mapping
                    else:
                        existing_img = next((img for img in combined_data['images'] if img['file_name'] == image['file_name']), None)
                        if existing_img:
                            image_id_map[image['id']] = existing_img['id']


                for annotation in data.get('annotations', []):
                    old_image_id = annotation['image_id']
                    old_category_id = annotation['category_id']

                    # Check if the annotation belongs to an image we included (or mapped)
                    # and refers to a category we included (or mapped)
                    if old_image_id in image_id_map and old_category_id in current_file_cat_id_map:
                        new_ann = annotation.copy() # Avoid modifying original
                        new_ann['id'] = next_annotation_id
                        new_ann['image_id'] = image_id_map[old_image_id]
                        new_ann['category_id'] = current_file_cat_id_map[old_category_id]
                        combined_data['annotations'].append(new_ann)
                        next_annotation_id += 1

            # Get output file path
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Combined JSON", "", "JSON Files (*.json)")
            if output_file:
                # Ensure .json extension
                if not output_file.lower().endswith('.json'):
                    output_file += '.json'

                with open(output_file, 'w', encoding='utf-8') as f: # Specify encoding
                    json.dump(combined_data, f, indent=2)
                QMessageBox.information(self, "Success", f"Combined JSON saved to {output_file}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while combining JSON files: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed error to console


    def show_centered(self, parent):
        # Geometry logic remains the same
        if parent:
            parent_geo = parent.geometry()
            self.move(parent_geo.center() - self.rect().center())
        self.show()

# Function to show the dialog (remains the same structure)
def show_coco_json_combiner(parent):
    dialog = COCOJSONCombinerDialog(parent)
    dialog.show_centered(parent) # Call show_centered
    return dialog # Return instance
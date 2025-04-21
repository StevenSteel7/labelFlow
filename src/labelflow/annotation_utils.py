# annotation_utils.py (Migrated to PyQt6)

# PyQt6 Imports
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

class AnnotationUtils:
    @staticmethod
    def update_annotation_list(self, image_name=None):
        """
        Updates the annotation list widget displayed in the UI.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
            image_name (str, optional): The specific image/slice name to load annotations for.
                                        Defaults to the currently viewed image/slice.
        """
        self.annotation_list.clear()
        current_name = image_name or self.current_slice or self.image_file_name
        annotations = self.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            # Use Qt.GlobalColor enum for colors
            color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
            for i, annotation in enumerate(class_annotations, start=1):
                # Assuming 'number' field is added/managed elsewhere, or use 'i'
                number = annotation.get('number', i) # Use actual number if available
                item_text = f"{class_name} - {number}"
                item = QListWidgetItem(item_text)
                # Use Qt.ItemDataRole enum for roles
                item.setData(Qt.ItemDataRole.UserRole, annotation)
                item.setForeground(color)
                self.annotation_list.addItem(item)

    @staticmethod
    def update_slice_list_colors(self):
        """
        Updates the colors of items in the slice list widget to indicate if they have annotations.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
        """
        for i in range(self.slice_list.count()):
            item = self.slice_list.item(i)
            slice_name = item.text()
            # Check if slice has any annotations across all classes
            has_annotations = slice_name in self.all_annotations and any(self.all_annotations[slice_name].values())

            if has_annotations:
                # Use Qt.GlobalColor enum for colors
                item.setForeground(QColor(Qt.GlobalColor.green))
            else:
                # Use Qt.GlobalColor enum for colors
                default_color = QColor(Qt.GlobalColor.white) if self.dark_mode else QColor(Qt.GlobalColor.black)
                item.setForeground(default_color)
        self.slice_list.repaint() # Ensure update is visible

    @staticmethod
    def update_annotation_list_colors(self, class_name=None, color=None):
        """
        Updates the text color of items in the annotation list, optionally for a specific class.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
            class_name (str, optional): The specific class name to update colors for.
                                        If None, updates colors for all items based on their class.
            color (QColor, optional): The specific color to apply if class_name is provided.
        """
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            # Use Qt.ItemDataRole enum for roles
            annotation = item.data(Qt.ItemDataRole.UserRole)
            if annotation: # Ensure data is valid
                item_class_name = annotation.get('category_name')
                if item_class_name:
                    if class_name is None or item_class_name == class_name:
                        # Use Qt.GlobalColor enum for default color
                        item_color = color if color else self.image_label.class_colors.get(item_class_name, QColor(Qt.GlobalColor.white))
                        item.setForeground(item_color)
        self.annotation_list.repaint() # Ensure update is visible

    @staticmethod
    def load_image_annotations(self):
        """
        Loads annotations for the currently selected image/slice into the image label.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
        """
        self.image_label.annotations.clear()
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            # Use deepcopy if modifications in image_label shouldn't affect all_annotations directly
            # For now, assuming copy is sufficient as per original code
            self.image_label.annotations = self.all_annotations[current_name].copy()
        self.image_label.update()

    @staticmethod
    def save_current_annotations(self):
        """
        Saves the annotations currently in the image label back to the main storage.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
        """
        current_name = self.current_slice or self.image_file_name
        if current_name:
            if self.image_label.annotations:
                # Save a copy to avoid direct modification issues
                self.all_annotations[current_name] = self.image_label.annotations.copy()
            elif current_name in self.all_annotations:
                # Remove if no annotations are left for this image/slice
                del self.all_annotations[current_name]
        AnnotationUtils.update_slice_list_colors(self) # Update colors after saving

    @staticmethod
    def add_annotation_to_list(self, annotation):
        """
        Adds a single annotation item to the annotation list widget.

        Args:
            self: The instance of the main window (e.g., ImageAnnotator).
            annotation (dict): The annotation dictionary to add.
        """
        class_name = annotation.get('category_name')
        if not class_name: return # Skip if no category name

        # Use Qt.GlobalColor enum for default color
        color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
        # Calculate number based on existing annotations for this class in the *current* view
        # This assumes numbering is relative to the current image/slice display
        current_annotations_for_class = self.image_label.annotations.get(class_name, [])
        number = annotation.get('number')
        if number is None: # Assign a number if not present
             existing_numbers = [ann.get('number', 0) for ann in current_annotations_for_class]
             number = max(existing_numbers + [0]) + 1
             annotation['number'] = number # Update the annotation dict itself

        item_text = f"{class_name} - {number}"
        item = QListWidgetItem(item_text)
        # Use Qt.ItemDataRole enum for roles
        item.setData(Qt.ItemDataRole.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)
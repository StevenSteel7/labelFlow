# project_details.py (Migrated to PyQt6 - Corrected)
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel,
                             QDialogButtonBox, QScrollArea, QWidget, QGroupBox, QGridLayout) # Added QGroupBox, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import os
from datetime import datetime
import traceback # Added for debugging potential errors

class ProjectDetailsDialog(QDialog):
    def __init__(self, parent=None, stats_dialog=None):
        super().__init__(parent)
        # Store the main application window reference safely
        # 'parent' in QDialog refers to the QWidget parent, not necessarily the main app window
        self.parent_window = parent
        self.stats_dialog = stats_dialog # The stats dialog instance (already created)
        self.setWindowTitle("Project Details")
        self.setModal(True)
        self.setMinimumSize(600, 800) # Set initial size

        # Store original notes to check for changes, access safely via parent_window
        self.original_notes = getattr(self.parent_window, 'project_notes', '')
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10) # Add some spacing

        # Scroll area for potentially long content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Use enum
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded) # Use enum
        scroll_content = QWidget() # Widget to hold the scrollable content
        scroll_layout = QVBoxLayout(scroll_content) # Set layout for the content widget
        scroll_layout.setSpacing(15) # Spacing within scroll area

        # Helper function to create bold labels
        def bold_label(text):
            label = QLabel(text)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            return label

        # Helper function to format datetime strings safely
        def format_datetime(date_string):
            if not date_string: return "N/A"
            try:
                # Attempt parsing standard ISO format, ensure input is string
                dt = datetime.fromisoformat(str(date_string))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                 # Fallback for potentially already formatted strings or other formats
                return str(date_string)

        # --- Project Metadata ---
        metadata_group = QGroupBox("Project Info") # Group related info
        metadata_layout = QGridLayout() # Use grid for alignment
        # Access parent_window attributes safely
        project_file_text = os.path.basename(getattr(self.parent_window, 'current_project_file', 'N/A'))
        creation_date_text = format_datetime(getattr(self.parent_window, 'project_creation_date', ''))
        last_modified_text = format_datetime(getattr(self.parent_window, 'last_modified', ''))

        metadata_layout.addWidget(bold_label("Project File:"), 0, 0, Qt.AlignmentFlag.AlignTop) # Align top
        metadata_layout.addWidget(QLabel(project_file_text), 0, 1)
        metadata_layout.addWidget(bold_label("Creation Date:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        metadata_layout.addWidget(QLabel(creation_date_text), 1, 1)
        metadata_layout.addWidget(bold_label("Last Modified:"), 2, 0, Qt.AlignmentFlag.AlignTop)
        metadata_layout.addWidget(QLabel(last_modified_text), 2, 1)
        metadata_layout.setColumnStretch(1, 1) # Allow value label to stretch
        metadata_group.setLayout(metadata_layout)
        scroll_layout.addWidget(metadata_group)

        # --- Image Information ---
        image_group = QGroupBox("Image Summary")
        image_layout = QVBoxLayout()
        image_layout.setSpacing(5) # Reduce spacing within image group
        all_images_list = getattr(self.parent_window, 'all_images', [])
        image_count = len(all_images_list)
        image_layout.addWidget(QLabel(f"<b>Total Images:</b> {image_count}"))

        # List image file names (limit display if too many)
        image_names = sorted([f"• {os.path.basename(img_info['file_name'])}" for img_info in all_images_list])

        max_images_to_list = 25 # Limit number shown
        if image_names:
             display_names = image_names[:max_images_to_list] + ["..."] if len(image_names) > max_images_to_list else image_names
             # Use RichText for bolding and line breaks
             image_list_label = QLabel("<b>Image Files:</b><br>" + "<br>".join(display_names))
             image_list_label.setTextFormat(Qt.TextFormat.RichText) # Use enum
             image_list_label.setWordWrap(True) # Word wrap might not work perfectly with forced breaks
             image_layout.addWidget(image_list_label)

        # Multi-dimensional image information
        image_slices_dict = getattr(self.parent_window, 'image_slices', {})
        multi_slice_images = [img for img in all_images_list if img.get('is_multi_slice', False)]
        if multi_slice_images:
            image_layout.addSpacing(5) # Add a little space
            image_layout.addWidget(QLabel(f"<b>Multi-dimensional Images:</b> {len(multi_slice_images)}"))
            multi_slice_text = ""
            for img in multi_slice_images[:max_images_to_list]: # Limit display
                base_name = os.path.splitext(img['file_name'])[0]
                # Get slice count from image_slices dict first
                slice_count = len(image_slices_dict.get(base_name, []))
                # Fallback to 'slices' key in image info
                if slice_count == 0: slice_count = len(img.get('slices', []))

                dims = img.get('dimensions', [])
                shape = img.get('shape', [])
                dims_str = f" (Dims: {','.join(map(str,dims))}, Shape: {'x'.join(map(str, shape))})" if dims and shape else ""
                multi_slice_text += f"• {os.path.basename(img['file_name'])}: {slice_count} slices{dims_str}<br>"
            if len(multi_slice_images) > max_images_to_list:
                 multi_slice_text += "...<br>"
            multi_slice_label = QLabel(multi_slice_text)
            multi_slice_label.setTextFormat(Qt.TextFormat.RichText) # Use enum
            image_layout.addWidget(multi_slice_label)

        image_group.setLayout(image_layout)
        scroll_layout.addWidget(image_group)

        # --- Annotation Information ---
        annotation_group = QGroupBox("Annotation Summary")
        annotation_layout = QVBoxLayout()
        annotation_layout.setSpacing(5)
        class_map = getattr(self.parent_window, 'class_mapping', {})
        class_names = sorted(list(class_map.keys()))
        annotation_layout.addWidget(QLabel(f"<b>Classes Defined:</b> {len(class_names)}"))
        if class_names:
             class_list_str = ", ".join(class_names)
             class_list_label = QLabel(f"<b>Classes:</b> {class_list_str}")
             class_list_label.setWordWrap(True)
             annotation_layout.addWidget(class_list_label)

        # Add annotation statistics if provided and valid
        if self.stats_dialog and hasattr(self.stats_dialog, 'text_browser'):
            annotation_layout.addSpacing(5)
            annotation_layout.addWidget(QLabel("<b>Annotation Statistics:</b>"))
            # Get stats text and format slightly using HTML line breaks
            stats_text = self.stats_dialog.text_browser.toPlainText()
            stats_label = QLabel(stats_text.replace('\n', '<br>'))
            stats_label.setTextFormat(Qt.TextFormat.RichText) # Use enum
            stats_label.setWordWrap(True)
            annotation_layout.addWidget(stats_label)

        annotation_group.setLayout(annotation_layout)
        scroll_layout.addWidget(annotation_group)

        # --- Project Notes ---
        notes_group = QGroupBox("Project Notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.original_notes) # Use original notes loaded initially
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        scroll_layout.addWidget(notes_group)

        scroll_layout.addStretch(1) # Add stretch to push content up

        # Set the content widget for the scroll area *AFTER* adding content and setting layout
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area) # Add scroll area to main dialog layout

        # --- Dialog Buttons ---
        # Use standard buttons for OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel) # Use enum
        button_box.accepted.connect(self.accept) # Built-in accept slot
        button_box.rejected.connect(self.reject) # Built-in reject slot
        layout.addWidget(button_box)

    def get_notes(self):
        """Returns the current text from the notes editor."""
        return self.notes_edit.toPlainText()

    def were_changes_made(self):
        """Checks if the notes text has changed from the original."""
        return self.get_notes() != self.original_notes

    # No need for separate accept/reject slots when using standard buttons
# image_label.py (Migrated to PyQt6)
"""
ImageLabel module for the Image Annotator application.

This module contains the ImageLabel class, which is responsible for
displaying the image and handling annotation interactions.

@LabelFlow
Mr. Steven Moses
"""

# PyQt6 Imports
from PyQt6.QtWidgets import QLabel, QApplication, QMessageBox
from PyQt6.QtGui import (QPainter, QPen, QColor, QFont, QPolygonF, QBrush,
                         QPixmap, QImage, QWheelEvent, QMouseEvent, QKeyEvent, QCursor) # Added QCursor
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QSize

# Other Imports
from PIL import Image
import os
import warnings
import cv2
import numpy as np
from shapely.geometry import Polygon, Point # Import Polygon and Point
import copy

warnings.filterwarnings("ignore", category=UserWarning)


class ImageLabel(QLabel):
    """
    A custom QLabel for displaying images and handling annotations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = {}
        self.current_annotation = []
        self.temp_point = None
        self.current_tool = None
        self.zoom_factor = 1.0
        self.class_colors = {}
        self.class_visibility = {}
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations = []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # Use enum
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.pan_start_pos = None
        self.main_window = None
        self.offset_x = 0
        self.offset_y = 0
        self.drawing_polygon = False
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.fill_opacity = 0.3
        self.drawing_rectangle = False
        self.current_rectangle = None
        self.bit_depth = None
        self.image_path = None
        self.dark_mode = False

        self.paint_mask = None
        self.eraser_mask = None
        self.temp_paint_mask = None
        self.is_painting = False
        self.temp_eraser_mask = None
        self.is_erasing = False
        self.cursor_pos = None

        #SAM
        self.sam_magic_wand_active = False
        self.sam_bbox = None
        self.drawing_sam_bbox = False
        self.temp_sam_prediction = None

        self.temp_annotations = []


    def set_main_window(self, main_window):
        self.main_window = main_window

    def set_dark_mode(self, is_dark):
        self.dark_mode = is_dark
        self.update()

    def setPixmap(self, pixmap):
        """Set the pixmap and update the scaled version."""
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def detect_bit_depth(self):
        """Detect and store the actual image bit depth using PIL."""
        if self.image_path and os.path.exists(self.image_path):
            try:
                with Image.open(self.image_path) as img:
                    mode_to_depth = {
                        '1': 1, 'L': 8, 'P': 8, 'I': 32, 'F': 32, # Basic modes
                        'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24,
                        'I;16': 16, 'I;16B': 16, 'I;16L': 16, # 16-bit grayscale variations
                        'BGR;16': 16*3, 'BGR;32': 32*3 # Less common OpenCV formats might appear
                    }
                    # Try to map common modes first
                    if img.mode in mode_to_depth:
                        self.bit_depth = mode_to_depth[img.mode]
                    elif hasattr(img, 'bits'): # Fallback for older PIL/Pillow
                        self.bit_depth = img.bits
                    elif img.mode.startswith('I;'): # Heuristic for integer types
                        try:
                            self.bit_depth = int(img.mode.split(';')[1])
                        except:
                            self.bit_depth = None # Unknown
                    elif img.mode.startswith('RGB;'): # Heuristic for RGB types
                         try:
                            self.bit_depth = int(img.mode.split(';')[1]) * 3
                         except:
                             self.bit_depth = 24 # Default RGB
                    else:
                         print(f"Warning: Unknown PIL image mode: {img.mode}. Bit depth detection might be inaccurate.")
                         self.bit_depth = None # Unknown

                    if self.main_window:
                        self.main_window.update_image_info()
            except Exception as e:
                print(f"Error detecting bit depth for {self.image_path}: {e}")
                self.bit_depth = None


    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            # Check if scaled size is valid
            if scaled_size.width() > 0 and scaled_size.height() > 0:
                 self.scaled_pixmap = self.original_pixmap.scaled(
                     scaled_size.width(),
                     scaled_size.height(),
                     Qt.AspectRatioMode.KeepAspectRatio, # Use enum
                     Qt.TransformationMode.SmoothTransformation # Use enum
                 )
                 super().setPixmap(self.scaled_pixmap)
                 self.setMinimumSize(self.scaled_pixmap.size())
                 self.update_offset()
            else:
                 print(f"Warning: Invalid scaled size calculated ({scaled_size.width()}x{scaled_size.height()}). Clearing pixmap.")
                 self.scaled_pixmap = None
                 super().setPixmap(QPixmap())
                 self.setMinimumSize(QSize(0, 0))
        else:
            self.scaled_pixmap = None
            super().setPixmap(QPixmap())
            self.setMinimumSize(QSize(0, 0))

    def update_offset(self):
        """Update the offset for centered image display."""
        if self.scaled_pixmap and not self.scaled_pixmap.isNull():
            self.offset_x = int((self.width() - self.scaled_pixmap.width()) / 2)
            self.offset_y = int((self.height() - self.scaled_pixmap.height()) / 2)
        else:
             self.offset_x = 0
             self.offset_y = 0

    def reset_annotation_state(self):
        """Reset the annotation state."""
        self.temp_point = None
        self.start_point = None
        self.end_point = None

    def clear_current_annotation(self):
        """Clear the current annotation."""
        self.current_annotation = []

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_offset()


    def start_painting(self, pos):
        if not self.original_pixmap or self.original_pixmap.isNull(): return
        if self.temp_paint_mask is None:
            self.temp_paint_mask = np.zeros((self.original_pixmap.height(), self.original_pixmap.width()), dtype=np.uint8)
        self.is_painting = True
        self.continue_painting(pos)

    def continue_painting(self, pos):
        if not self.is_painting or self.temp_paint_mask is None:
            return
        brush_size = self.main_window.paint_brush_size
        # Clamp coordinates to be within mask bounds
        x = int(max(0, min(pos[0], self.temp_paint_mask.shape[1] - 1)))
        y = int(max(0, min(pos[1], self.temp_paint_mask.shape[0] - 1)))
        cv2.circle(self.temp_paint_mask, (x, y), brush_size, 255, -1)
        self.update()

    def finish_painting(self):
        if not self.is_painting:
            return
        self.is_painting = False
        # Don't commit the annotation yet, just keep the temp_paint_mask


    def commit_paint_annotation(self):
        if self.temp_paint_mask is not None and self.main_window.current_class:
            class_name = self.main_window.current_class
            contours, _ = cv2.findContours(self.temp_paint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            added_annotations = [] # Keep track of added annotations for add_annotation_to_list
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Minimum area threshold
                    segmentation = contour.flatten().tolist()
                    if len(segmentation) >= 6: # Need at least 3 points for a polygon
                        new_annotation = {
                            "segmentation": segmentation,
                            "category_id": self.main_window.class_mapping[class_name],
                            "category_name": class_name,
                            "type": "polygon" # Painted area is a polygon
                        }
                        self.annotations.setdefault(class_name, []).append(new_annotation)
                        added_annotations.append(new_annotation)

            self.temp_paint_mask = None
            if added_annotations:
                 self.main_window.save_current_annotations()
                 self.main_window.update_slice_list_colors()
                 # Add to list after saving/updating other state
                 for ann in added_annotations:
                      self.main_window.add_annotation_to_list(ann)
            self.update()


    def discard_paint_annotation(self):
        self.temp_paint_mask = None
        self.update()


    def start_erasing(self, pos):
        if not self.original_pixmap or self.original_pixmap.isNull(): return
        if self.temp_eraser_mask is None:
            self.temp_eraser_mask = np.zeros((self.original_pixmap.height(), self.original_pixmap.width()), dtype=np.uint8)
        self.is_erasing = True
        self.continue_erasing(pos)

    def continue_erasing(self, pos):
        if not self.is_erasing or self.temp_eraser_mask is None:
            return
        eraser_size = self.main_window.eraser_size
         # Clamp coordinates to be within mask bounds
        x = int(max(0, min(pos[0], self.temp_eraser_mask.shape[1] - 1)))
        y = int(max(0, min(pos[1], self.temp_eraser_mask.shape[0] - 1)))
        cv2.circle(self.temp_eraser_mask, (x, y), eraser_size, 255, -1)
        self.update()

    def finish_erasing(self):
        if not self.is_erasing:
            return
        self.is_erasing = False
        # Don't commit the eraser changes yet, just keep the temp_eraser_mask

    def commit_eraser_changes(self):
        if self.temp_eraser_mask is not None:
            eraser_mask = self.temp_eraser_mask.astype(bool)
            current_name = self.main_window.current_slice or self.main_window.image_file_name
            annotations_changed = False
            original_annotations_copy = copy.deepcopy(self.annotations) # Work on a copy

            for class_name, annotations in original_annotations_copy.items():
                updated_annotations = []
                max_number = max([ann.get('number', 0) for ann in annotations] + [0])
                for annotation in annotations:
                    if "segmentation" in annotation:
                        try:
                             points = np.array(annotation["segmentation"]).reshape(-1, 2).astype(np.int32) # Use int32 for fillPoly
                             # Ensure mask has same dimensions as image
                             mask_height, mask_width = self.temp_eraser_mask.shape[:2]
                             mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
                             cv2.fillPoly(mask, [points], 255)
                             mask = mask.astype(bool)
                             mask[eraser_mask] = False # Apply eraser
                             contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                             if not contours: # If erasing removed the whole shape
                                 annotations_changed = True
                                 continue # Skip adding this annotation back

                             new_pieces = []
                             for i, contour in enumerate(contours):
                                 if cv2.contourArea(contour) > 10:  # Minimum area threshold
                                     new_segmentation = contour.flatten().tolist()
                                     if len(new_segmentation) >= 6:
                                         new_annotation = annotation.copy()
                                         new_annotation["segmentation"] = new_segmentation
                                         # Assign number - keep original for first piece, generate new for others
                                         if not new_pieces: # First valid piece
                                             new_annotation["number"] = annotation.get("number", max_number + 1)
                                         else:
                                             max_number += 1
                                             new_annotation["number"] = max_number
                                         new_pieces.append(new_annotation)

                             if new_pieces:
                                 updated_annotations.extend(new_pieces)
                                 if len(new_pieces) != 1 or new_pieces[0]["segmentation"] != annotation["segmentation"]:
                                     annotations_changed = True # Mark as changed if shape altered or split
                             else: # No valid pieces left
                                 annotations_changed = True


                        except Exception as e:
                             print(f"Error processing annotation during erase: {e}")
                             updated_annotations.append(annotation) # Keep original if error
                    else:
                        updated_annotations.append(annotation) # Keep non-polygon annotations

                if updated_annotations or class_name in self.annotations: # Only update if there's something new or old
                    self.annotations[class_name] = updated_annotations # Update the actual annotations dict
            # Remove empty class lists
            keys_to_del = [k for k, v in self.annotations.items() if not v]
            for k in keys_to_del:
                del self.annotations[k]

            self.temp_eraser_mask = None

            # Update the all_annotations dictionary in the main window
            if current_name: # Ensure we have a key
                self.main_window.all_annotations[current_name] = self.annotations.copy()

            # Call update_annotation_list directly
            self.main_window.update_annotation_list()

            self.main_window.save_current_annotations()
            self.main_window.update_slice_list_colors()
            self.update()

            #print(f"Eraser changes committed. Annotations changed: {annotations_changed}")
            #print(f"Current annotations: {self.annotations}")

    def discard_eraser_changes(self):
        self.temp_eraser_mask = None
        self.update()


    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap and not self.scaled_pixmap.isNull(): # Check scaled_pixmap is valid
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing) # Use enum

            # Draw the image
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), self.scaled_pixmap)

            # Draw annotations
            self.draw_annotations(painter)

            # Draw other elements
            if self.editing_polygon:
                self.draw_editing_polygon(painter)

            if self.drawing_rectangle and self.current_rectangle:
                self.draw_current_rectangle(painter)

            if self.sam_magic_wand_active and self.sam_bbox:
                self.draw_sam_bbox(painter)

            # Draw temporary paint mask
            if self.temp_paint_mask is not None:
                self.draw_temp_paint_mask(painter)

            # Draw temporary eraser mask
            if self.temp_eraser_mask is not None:
                self.draw_temp_eraser_mask(painter)

            # Draw brush/eraser size indicator
            self.draw_tool_size_indicator(painter)

            # Draw temporary YOLO predictions
            if self.temp_annotations:
                self.draw_temp_annotations(painter)

            painter.end()

    def draw_temp_annotations(self, painter):
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        pen_width = max(1.0, 2.0 / self.zoom_factor) # Ensure pen width is at least 1 pixel

        for annotation in self.temp_annotations:
            if not self.main_window.is_class_visible(annotation['category_name']):
                continue

            color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(color, pen_width, Qt.PenStyle.DashLine)) # Use enum
            painter.setBrush(QBrush(color))

            label = f"{annotation['category_name'][5:]} {annotation['score']:.2f}" # Remove "Temp-"

            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                rect = QRectF(x, y, w, h)
                painter.drawRect(rect)
                # Draw label near top-left corner of bbox
                painter.setFont(QFont("Arial", int(max(6, 12 / self.zoom_factor)))) # Min font size 6
                painter.setPen(Qt.GlobalColor.black) # Use enum
                painter.drawText(QPointF(rect.left(), rect.top() - 5), label)
            elif "segmentation" in annotation:
                points = [QPointF(float(px), float(py)) for px, py in zip(annotation["segmentation"][0::2], annotation["segmentation"][1::2])]
                if len(points) > 2:
                    painter.drawPolygon(QPolygonF(points))
                    # Draw label at centroid
                    centroid = self.calculate_centroid(points)
                    if centroid:
                        painter.setFont(QFont("Arial", int(max(6, 12 / self.zoom_factor))))
                        painter.setPen(Qt.GlobalColor.black) # Use enum
                        painter.drawText(centroid, label)

        painter.restore()


    def accept_temp_annotations(self):
        added_annotations = []
        for annotation in self.temp_annotations:
            # Check visibility before accepting
            if not self.main_window.is_class_visible(annotation['category_name']):
                 continue

            class_name = annotation['category_name'][5:] # Remove "Temp-" prefix

            # Check if the permanent class exists, if not, add it
            if class_name not in self.main_window.class_mapping:
                 # Find the color from the temp class
                 temp_color = self.image_label.class_colors.get(annotation['category_name'], QColor(Qt.GlobalColor.gray)) # Use enum
                 self.main_window.add_class(class_name, temp_color)

            if class_name not in self.annotations:
                self.annotations[class_name] = []

            # Prepare final annotation
            final_ann = annotation.copy()
            final_ann['category_name'] = class_name
            final_ann['category_id'] = self.main_window.class_mapping[class_name]
            if 'temp' in final_ann: del final_ann['temp']
            if 'score' in final_ann: del final_ann['score']

            self.annotations[class_name].append(final_ann)
            added_annotations.append(final_ann) # Add to list for adding to UI list

        # Clear temporary storage and remove temp classes from UI state
        temp_class_names_to_remove = list(self.temp_annotations[0]['category_name'] for ann in self.temp_annotations if 'category_name' in ann) # Get all temp names
        self.temp_annotations.clear()

        for temp_name in set(temp_class_names_to_remove): # Use set for unique names
            if temp_name in self.image_label.annotations:
                del self.image_label.annotations[temp_name]
            if temp_name in self.image_label.class_colors:
                del self.image_label.class_colors[temp_name]

        # Update main storage and UI list
        if added_annotations:
            current_name = self.main_window.current_slice or self.main_window.image_file_name
            if current_name:
                self.main_window.all_annotations[current_name] = self.annotations.copy()

            self.main_window.save_current_annotations()
            self.main_window.update_slice_list_colors()
            self.main_window.renumber_annotations() # Renumber after adding
            self.main_window.update_annotation_list() # Update UI list

        self.main_window.update_class_list() # Remove temp classes from list
        self.update()


    def discard_temp_annotations(self):
         # Get names before clearing
        temp_class_names_to_remove = list(set(ann['category_name'] for ann in self.temp_annotations if 'category_name' in ann))
        self.temp_annotations.clear()
        # Remove class colors and potentially from display annotations if added prematurely
        for temp_name in temp_class_names_to_remove:
            if temp_name in self.image_label.annotations:
                del self.image_label.annotations[temp_name]
            if temp_name in self.image_label.class_colors:
                del self.image_label.class_colors[temp_name]
        self.main_window.update_class_list() # Update the list widget
        self.update()


    def draw_temp_paint_mask(self, painter):
        if self.temp_paint_mask is not None:
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            mask_image = QImage(self.temp_paint_mask.data, self.temp_paint_mask.shape[1], self.temp_paint_mask.shape[0], self.temp_paint_mask.strides[0], QImage.Format.Format_Grayscale8) # Use enum
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.setOpacity(1.0)

            painter.restore()


    def draw_temp_eraser_mask(self, painter):
        if self.temp_eraser_mask is not None:
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            mask_image = QImage(self.temp_eraser_mask.data, self.temp_eraser_mask.shape[1], self.temp_eraser_mask.shape[0], self.temp_eraser_mask.strides[0], QImage.Format.Format_Grayscale8) # Use enum
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.setOpacity(1.0)

            painter.restore()


    def draw_tool_size_indicator(self, painter):
        if self.current_tool in ["paint_brush", "eraser"] and self.cursor_pos is not None:
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            if self.current_tool == "paint_brush":
                size = self.main_window.paint_brush_size
                color = QColor(255, 0, 0, 128)  # Semi-transparent red
            else:  # eraser
                size = self.main_window.eraser_size
                color = QColor(0, 0, 255, 128)  # Semi-transparent blue

            pen_width = max(1.0, 1.0 / self.zoom_factor) # Ensure outline is visible

            # Draw filled circle with lower opacity
            painter.setOpacity(0.3)
            painter.setPen(Qt.PenStyle.NoPen) # Use enum
            painter.setBrush(color)
            painter.drawEllipse(QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size)

            # Draw circle outline with full opacity
            painter.setOpacity(1.0)
            painter.setPen(QPen(color.darker(150), pen_width, Qt.PenStyle.SolidLine)) # Use enum
            painter.setBrush(Qt.BrushStyle.NoBrush) # Use enum
            painter.drawEllipse(QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size)

            # Draw size text
            painter.resetTransform()
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.black)) # Use enum

            # Convert cursor position back to screen coordinates
            screen_x = self.cursor_pos[0] * self.zoom_factor + self.offset_x
            screen_y = self.cursor_pos[1] * self.zoom_factor + self.offset_y

            # Position text near the circle
            text_rect = QRectF(screen_x + (size * self.zoom_factor) + 5,
                              screen_y - 10, # Adjusted Y position
                              100, 20)

            text = f"Size: {size}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text) # Use enum

            painter.restore()


    def draw_paint_mask(self, painter):
        # This seems related to older/unused paint functionality - keep if needed
        if self.paint_mask is not None:
            mask_image = QImage(self.paint_mask.data, self.paint_mask.shape[1], self.paint_mask.shape[0], self.paint_mask.strides[0], QImage.Format.Format_Grayscale8) # Use enum
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            if self.scaled_pixmap and not self.scaled_pixmap.isNull():
                painter.drawPixmap(self.offset_x, self.offset_y, mask_pixmap.scaled(self.scaled_pixmap.size()))
            painter.setOpacity(1.0)

    def draw_eraser_mask(self, painter):
         # This seems related to older/unused eraser functionality - keep if needed
        if self.eraser_mask is not None:
            mask_image = QImage(self.eraser_mask.data, self.eraser_mask.shape[1], self.eraser_mask.shape[0], self.eraser_mask.strides[0], QImage.Format.Format_Grayscale8) # Use enum
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            if self.scaled_pixmap and not self.scaled_pixmap.isNull():
                painter.drawPixmap(self.offset_x, self.offset_y, mask_pixmap.scaled(self.scaled_pixmap.size()))
            painter.setOpacity(1.0)


    def draw_sam_bbox(self, painter):
        if not self.sam_bbox: return
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        pen_width = max(1.0, 2.0 / self.zoom_factor)
        painter.setPen(QPen(Qt.GlobalColor.red, pen_width, Qt.PenStyle.SolidLine)) # Use enums
        x1, y1, x2, y2 = self.sam_bbox
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        painter.restore()

    def clear_temp_sam_prediction(self):
        self.temp_sam_prediction = None
        self.update()

    def check_unsaved_changes(self):
        if self.temp_paint_mask is not None or self.temp_eraser_mask is not None:
            reply = QMessageBox.question(
                self.main_window, 'Unsaved Changes',
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel # Use enum
            )
            if reply == QMessageBox.StandardButton.Yes: # Use enum
                if self.temp_paint_mask is not None:
                    self.commit_paint_annotation()
                if self.temp_eraser_mask is not None:
                    self.commit_eraser_changes()
                return True
            elif reply == QMessageBox.StandardButton.No: # Use enum
                self.discard_paint_annotation()
                self.discard_eraser_changes()
                return True
            else:  # Cancel
                return False
        return True  # No unsaved changes

    def clear(self):
        super().clear()
        self.annotations.clear()
        self.current_annotation.clear()
        self.temp_point = None
        self.current_tool = None
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations.clear()
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.current_rectangle = None
        self.sam_bbox = None
        self.temp_sam_prediction = None
        self.temp_annotations.clear() # Clear temp YOLO annotations too
        self.temp_paint_mask = None
        self.temp_eraser_mask = None
        self.update()


    def set_class_visibility(self, class_name, is_visible):
        self.class_visibility[class_name] = is_visible

    def draw_annotations(self, painter):
        """Draw all annotations on the image."""
        if not self.original_pixmap or self.original_pixmap.isNull():
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        pen_width = max(1.0, 2.0 / self.zoom_factor) # Ensure pen width is at least 1 pixel
        font_size = int(max(6, 12 / self.zoom_factor)) # Ensure font size is at least 6pt

        for class_name, class_annotations in self.annotations.items():
            # Check visibility based on main window's class list state
            if not self.main_window.is_class_visible(class_name):
                continue

            color = self.class_colors.get(class_name, QColor(Qt.GlobalColor.white)) # Use enum
            for annotation in class_annotations:
                is_highlighted = annotation in self.highlighted_annotations
                is_temp = annotation.get('temp', False) # Check for temporary annotations

                if is_temp: # Style for temporary annotations (e.g., from YOLO)
                    border_color = QColor(255, 165, 0) # Orange
                    fill_color = QColor(255, 165, 0, 128) # Semi-transparent orange
                    pen_style = Qt.PenStyle.DashLine # Use enum
                    text_color = Qt.GlobalColor.black # Use enum
                elif is_highlighted:
                    border_color = QColor(Qt.GlobalColor.red) # Use enum
                    fill_color = QColor(Qt.GlobalColor.red) # Use enum
                    pen_style = Qt.PenStyle.SolidLine # Use enum
                    text_color = Qt.GlobalColor.white if self.dark_mode else Qt.GlobalColor.black # Use enum
                else:
                    border_color = color
                    fill_color = QColor(color)
                    pen_style = Qt.PenStyle.SolidLine # Use enum
                    text_color = Qt.GlobalColor.white if self.dark_mode else Qt.GlobalColor.black # Use enum

                fill_color.setAlphaF(self.fill_opacity)

                painter.setPen(QPen(border_color, pen_width, pen_style))
                painter.setBrush(QBrush(fill_color))

                label_text = f"{annotation.get('category_name', class_name)} {annotation.get('number', '')}".strip()
                if is_temp and 'score' in annotation:
                    label_text = f"{label_text} {annotation['score']:.2f}"

                if "segmentation" in annotation:
                    segmentation = annotation["segmentation"]
                    if isinstance(segmentation, list) and len(segmentation) >= 6:
                         points = [QPointF(float(x), float(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
                         painter.drawPolygon(QPolygonF(points))
                         # Draw centroid and label
                         centroid = self.calculate_centroid(points)
                         if centroid:
                             painter.setFont(QFont("Arial", font_size))
                             painter.setPen(QPen(text_color)) # No width needed for text pen
                             painter.drawText(centroid, label_text)

                elif "bbox" in annotation:
                    x, y, width, height = annotation["bbox"]
                    rect = QRectF(x, y, width, height)
                    painter.drawRect(rect)
                    painter.setFont(QFont("Arial", font_size))
                    painter.setPen(QPen(text_color))
                    painter.drawText(QPointF(rect.left(), rect.top() - 5), label_text) # Position above bbox

        # Draw the polygon currently being drawn
        if self.drawing_polygon and self.current_annotation:
            current_draw_color = QColor(Qt.GlobalColor.red) # Use enum
            painter.setPen(QPen(current_draw_color, pen_width, Qt.PenStyle.SolidLine)) # Use enum
            points = [QPointF(float(x), float(y)) for x, y in self.current_annotation]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            painter.setBrush(current_draw_color)
            for point in points:
                painter.drawEllipse(point, 5 / self.zoom_factor, 5 / self.zoom_factor)
            if self.temp_point: # Draw line to cursor
                painter.drawLine(points[-1], QPointF(float(self.temp_point[0]), float(self.temp_point[1])))

        # Draw temporary SAM prediction (if any)
        if self.temp_sam_prediction:
            temp_color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(temp_color, pen_width, Qt.PenStyle.DashLine)) # Use enum
            painter.setBrush(QBrush(temp_color))

            segmentation = self.temp_sam_prediction["segmentation"]
            points = [QPointF(float(x), float(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
            if len(points) > 2:
                painter.drawPolygon(QPolygonF(points))
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.setFont(QFont("Arial", font_size))
                    painter.setPen(QPen(Qt.GlobalColor.black)) # Use enum
                    painter.drawText(centroid, f"SAM: {self.temp_sam_prediction['score']:.2f}")

        painter.restore()


    def draw_current_rectangle(self, painter):
        """Draw the current rectangle being created."""
        if not self.current_rectangle:
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        x1, y1, x2, y2 = self.current_rectangle
        color = self.class_colors.get(self.main_window.current_class, QColor(Qt.GlobalColor.red)) # Use enum
        pen_width = max(1.0, 2.0 / self.zoom_factor)
        painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine)) # Use enum
        # Ensure width and height are positive for QRectF
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        painter.drawRect(rect)

        painter.restore()

    def get_rectangle_from_points(self):
        """Get rectangle coordinates from start and end points."""
        if not self.start_point or not self.end_point:
            return None
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        return [x1, y1, x2, y2] # Keep as x1,y1,x2,y2 for drawing, convert to x,y,w,h on finish

    def draw_editing_polygon(self, painter):
        """Draw the polygon being edited."""
        if not self.editing_polygon or "segmentation" not in self.editing_polygon:
            return
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        points = [QPointF(float(x), float(y)) for x, y in zip(self.editing_polygon["segmentation"][0::2], self.editing_polygon["segmentation"][1::2])]
        color = self.class_colors.get(self.editing_polygon["category_name"], QColor(Qt.GlobalColor.white)) # Use enum
        fill_color = QColor(color)
        fill_color.setAlphaF(self.fill_opacity)
        pen_width = max(1.0, 2.0 / self.zoom_factor)

        painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine)) # Use enum
        painter.setBrush(QBrush(fill_color))
        if len(points) > 2:
             painter.drawPolygon(QPolygonF(points))

        # Draw editing handles
        handle_radius = max(2.0, 5.0 / self.zoom_factor) # Ensure handles are visible
        for i, point in enumerate(points):
            if i == self.hover_point_index:
                painter.setBrush(QColor(Qt.GlobalColor.red)) # Use enum
            else:
                painter.setBrush(QColor(Qt.GlobalColor.green)) # Use enum
            painter.drawEllipse(point, handle_radius, handle_radius)

        painter.restore()

    def calculate_centroid(self, points):
        """Calculate the centroid of a polygon."""
        if not points or len(points) < 3:
            return None
        # Use shoelace formula for area and centroid calculation
        polygon = Polygon([(p.x(), p.y()) for p in points])
        if polygon.is_valid and polygon.area > 0:
            centroid = polygon.centroid
            return QPointF(centroid.x, centroid.y)
        else: # Fallback for invalid polygons or lines
            x_coords = [point.x() for point in points]
            y_coords = [point.y() for point in points]
            return QPointF(sum(x_coords) / len(points), sum(y_coords) / len(points))


    def set_zoom(self, zoom_factor):
        """Set the zoom factor and update the display."""
        self.zoom_factor = zoom_factor
        self.update_scaled_pixmap()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        # In PyQt6, use pixelDelta() or angleDelta()
        # angleDelta() is usually preferred for traditional mouse wheels
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier: # Use enum
            delta = event.angleDelta().y()
            if delta > 0:
                self.main_window.zoom_in()
            elif delta < 0: # Check for negative delta
                self.main_window.zoom_out()
            event.accept()
        else:
            # Allow vertical scrolling if image is larger than viewport
            scrollbar = self.main_window.scroll_area.verticalScrollBar()
            if scrollbar.maximum() > 0:
                 delta = event.angleDelta().y()
                 scrollbar.setValue(scrollbar.value() - delta)
                 event.accept()
            else: # Pass to parent if no scrolling needed
                 super().wheelEvent(event)


    def mousePressEvent(self, event: QMouseEvent):
        if not self.original_pixmap or self.original_pixmap.isNull():
            return

        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.button() == Qt.MouseButton.LeftButton: # Use enums
            self.pan_start_pos = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor)) # Use QCursor and enum
            event.accept()
        else:
            pos = self.get_image_coordinates(event.pos())
            # Ensure pos is within image bounds
            img_w = self.original_pixmap.width()
            img_h = self.original_pixmap.height()
            if not (0 <= pos[0] < img_w and 0 <= pos[1] < img_h):
                print("Click outside image bounds")
                return # Ignore clicks outside the image area

            if event.button() == Qt.MouseButton.LeftButton: # Use enum
                if self.sam_magic_wand_active:
                    self.sam_bbox = [pos[0], pos[1], pos[0], pos[1]]
                    self.drawing_sam_bbox = True
                elif self.editing_polygon:
                    self.handle_editing_click(pos, event)
                elif self.current_tool == "polygon":
                    if not self.drawing_polygon:
                        # Discard any previous temp YOLO annotations when starting a new manual one
                        if self.temp_annotations: self.discard_temp_annotations()
                        self.drawing_polygon = True
                        self.current_annotation = []
                    self.current_annotation.append(pos)
                elif self.current_tool == "rectangle":
                    # Discard any previous temp YOLO annotations when starting a new manual one
                    if self.temp_annotations: self.discard_temp_annotations()
                    self.start_point = pos
                    self.end_point = pos
                    self.drawing_rectangle = True
                    self.current_rectangle = None # Reset current rectangle display
                elif self.current_tool == "paint_brush":
                    # Discard any previous temp YOLO annotations when starting a new manual one
                    if self.temp_annotations: self.discard_temp_annotations()
                    self.start_painting(pos)
                elif self.current_tool == "eraser":
                    self.start_erasing(pos)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
        current_pos_screen = event.pos()
        self.cursor_pos = self.get_image_coordinates(current_pos_screen) # Update cursor pos in image coords

        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.buttons() == Qt.MouseButton.LeftButton: # Use enums
            if self.pan_start_pos:
                delta = current_pos_screen - self.pan_start_pos
                scrollbar_h = self.main_window.scroll_area.horizontalScrollBar()
                scrollbar_v = self.main_window.scroll_area.verticalScrollBar()
                scrollbar_h.setValue(scrollbar_h.value() - delta.x())
                scrollbar_v.setValue(scrollbar_v.value() - delta.y())
                self.pan_start_pos = current_pos_screen
            event.accept()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor)) # Ensure cursor stays closed hand
        else:
             # Reset cursor if not panning
             if self.current_tool == "sam_magic_wand":
                  self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
             elif self.current_tool in ["paint_brush", "eraser"]:
                  self.setCursor(Qt.CursorShape.BlankCursor) # Hide standard cursor when drawing indicator
             else:
                  self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

             pos = self.cursor_pos # Use updated image coords
             # Clamp pos to image bounds for drawing operations
             img_w = self.original_pixmap.width()
             img_h = self.original_pixmap.height()
             clamped_pos = (max(0, min(pos[0], img_w - 1)), max(0, min(pos[1], img_h - 1)))

             if self.sam_magic_wand_active and self.drawing_sam_bbox:
                 self.sam_bbox[2] = clamped_pos[0]
                 self.sam_bbox[3] = clamped_pos[1]
             elif self.editing_polygon:
                 self.handle_editing_move(clamped_pos)
             elif self.current_tool == "polygon" and self.drawing_polygon and self.current_annotation:
                 self.temp_point = clamped_pos
             elif self.current_tool == "rectangle" and self.drawing_rectangle:
                 self.end_point = clamped_pos
                 self.current_rectangle = self.get_rectangle_from_points()
             elif self.current_tool == "paint_brush" and event.buttons() == Qt.MouseButton.LeftButton: # Use enum
                 self.continue_painting(clamped_pos)
             elif self.current_tool == "eraser" and event.buttons() == Qt.MouseButton.LeftButton: # Use enum
                 self.continue_erasing(clamped_pos)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.original_pixmap or self.original_pixmap.isNull():
            return

        # Reset cursor after panning stops
        if self.pan_start_pos and event.button() == Qt.MouseButton.LeftButton: # Check button released was left
            self.pan_start_pos = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)) # Reset cursor
            event.accept()
        # Check if Control was pressed during release (might happen if released outside widget)
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.button() == Qt.MouseButton.LeftButton:
             self.pan_start_pos = None
             self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)) # Reset cursor
             event.accept()
        else:
            pos = self.get_image_coordinates(event.pos())
             # Clamp pos to image bounds
            img_w = self.original_pixmap.width()
            img_h = self.original_pixmap.height()
            clamped_pos = (max(0, min(pos[0], img_w - 1)), max(0, min(pos[1], img_h - 1)))

            if event.button() == Qt.MouseButton.LeftButton: # Use enum
                if self.sam_magic_wand_active and self.drawing_sam_bbox:
                    self.sam_bbox[2] = clamped_pos[0]
                    self.sam_bbox[3] = clamped_pos[1]
                    self.drawing_sam_bbox = False
                    if abs(self.sam_bbox[0]-self.sam_bbox[2]) > 3 and abs(self.sam_bbox[1]-self.sam_bbox[3]) > 3: # Min bbox size
                        self.main_window.apply_sam_prediction()
                    else:
                        self.sam_bbox = None # Discard tiny bbox
                elif self.editing_polygon:
                    if self.editing_point_index is not None:
                         # Finalize point move
                         self.editing_polygon["segmentation"][self.editing_point_index*2] = clamped_pos[0]
                         self.editing_polygon["segmentation"][self.editing_point_index*2+1] = clamped_pos[1]
                         # Recalculate bbox/area if needed
                         if 'bbox' in self.editing_polygon:
                              self.editing_polygon['bbox'] = self.main_window.utils.calculate_bbox(self.editing_polygon['segmentation'])
                         if 'area' in self.editing_polygon:
                              self.editing_polygon['area'] = self.main_window.utils.calculate_area(self.editing_polygon)
                         self.main_window.update_annotation_list() # Update list display
                         self.main_window.save_current_annotations() # Save change
                    self.editing_point_index = None # Stop dragging
                elif self.current_tool == "rectangle" and self.drawing_rectangle:
                    self.drawing_rectangle = False
                    self.end_point = clamped_pos # Set final point
                    self.current_rectangle = self.get_rectangle_from_points() # Final rectangle coords
                    # Check if rectangle is valid (non-zero size)
                    if self.current_rectangle and abs(self.current_rectangle[0]-self.current_rectangle[2]) > 1 and abs(self.current_rectangle[1]-self.current_rectangle[3]) > 1:
                        self.main_window.finish_rectangle()
                    else: # Discard tiny rectangle
                         self.start_point = None
                         self.end_point = None
                         self.current_rectangle = None
                elif self.current_tool == "paint_brush":
                    self.finish_painting()
                elif self.current_tool == "eraser":
                    self.finish_erasing()
        self.update()


    def mouseDoubleClickEvent(self, event):
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
        pos = self.get_image_coordinates(event.pos())
         # Clamp pos to image bounds
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        if not (0 <= pos[0] < img_w and 0 <= pos[1] < img_h): return

        if event.button() == Qt.MouseButton.LeftButton: # Use enum
            if self.drawing_polygon and len(self.current_annotation) > 2:
                # Check if double-click is close to the start point to close polygon
                if self.distance(pos, self.current_annotation[0]) < 10 / self.zoom_factor:
                    self.finish_polygon()
                else:
                    # Add point on double-click if not closing
                    self.current_annotation.append(pos)

            elif not self.editing_polygon: # Don't start edit on double click if already editing
                self.clear_current_annotation() # Clear any partial polygon
                annotation = self.start_polygon_edit(pos)
                if annotation:
                    self.main_window.select_annotation_in_list(annotation)
        self.update()

    def get_image_coordinates(self, pos):
        """Converts viewport coordinates to image coordinates."""
        if not self.scaled_pixmap or self.scaled_pixmap.isNull() or self.zoom_factor == 0:
            return (0, 0)
        x = (pos.x() - self.offset_x) / self.zoom_factor
        y = (pos.y() - self.offset_y) / self.zoom_factor
        return (x, y) # Return float for precision, convert to int when needed for indexing


    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter: # Use enum
            if self.temp_annotations:
                self.accept_temp_annotations()
            elif self.temp_sam_prediction:
                self.main_window.accept_sam_prediction()
            elif self.editing_polygon:
                self.exit_editing_mode() # Use dedicated method
                self.main_window.enable_tools()
                self.main_window.update_annotation_list()
            elif self.current_tool == "polygon" and self.drawing_polygon:
                self.finish_polygon()
            elif self.current_tool == "paint_brush" and self.temp_paint_mask is not None:
                self.commit_paint_annotation()
            elif self.current_tool == "eraser" and self.temp_eraser_mask is not None:
                self.commit_eraser_changes()
            else:
                super().keyPressEvent(event) # Pass up if not handled
        elif key == Qt.Key.Key_Escape: # Use enum
            if self.temp_annotations:
                self.discard_temp_annotations()
            elif self.sam_magic_wand_active:
                self.sam_bbox = None
                self.drawing_sam_bbox = False
                self.clear_temp_sam_prediction()
            elif self.editing_polygon:
                self.exit_editing_mode() # Use dedicated method
                self.main_window.enable_tools()
            elif self.drawing_polygon: # Cancel current polygon drawing
                 self.cancel_current_annotation()
            elif self.drawing_rectangle: # Cancel current rectangle drawing
                 self.drawing_rectangle = False
                 self.start_point = None
                 self.end_point = None
                 self.current_rectangle = None
            elif self.current_tool == "paint_brush" and self.temp_paint_mask is not None:
                self.discard_paint_annotation()
            elif self.current_tool == "eraser" and self.temp_eraser_mask is not None:
                self.discard_eraser_changes()
            else:
                super().keyPressEvent(event) # Pass up if not handled

        elif key == Qt.Key.Key_Delete: # Use enum
            if self.editing_polygon and self.main_window.annotation_list.selectedItems():
                self.main_window.delete_selected_annotations() # Use main window's method
                self.exit_editing_mode() # Exit edit mode after deletion
                self.main_window.enable_tools()
            else:
                 super().keyPressEvent(event)

        elif key == Qt.Key.Key_Minus: # Use enum
            if self.current_tool == "paint_brush":
                self.main_window.paint_brush_size = max(1, self.main_window.paint_brush_size - 1)
                print(f"Paint brush size: {self.main_window.paint_brush_size}")
            elif self.current_tool == "eraser":
                self.main_window.eraser_size = max(1, self.main_window.eraser_size - 1)
                print(f"Eraser size: {self.main_window.eraser_size}")
        elif key == Qt.Key.Key_Equal or key == Qt.Key.Key_Plus: # Use enum (+ is often Shift+=)
            if self.current_tool == "paint_brush":
                self.main_window.paint_brush_size += 1
                print(f"Paint brush size: {self.main_window.paint_brush_size}")
            elif self.current_tool == "eraser":
                self.main_window.eraser_size += 1
                print(f"Eraser size: {self.main_window.eraser_size}")
        else:
             super().keyPressEvent(event) # Pass unhandled keys up

        self.update()


    def cancel_current_annotation(self):
        """Cancel the current annotation being created."""
        if self.current_tool == "polygon" and self.current_annotation:
            self.current_annotation = []
            self.temp_point = None
            self.drawing_polygon = False
            self.update()


    def finish_current_annotation(self):
        """Finish the current annotation being created."""
        # This seems deprecated by finish_polygon / finish_rectangle
        # if self.current_tool == "polygon" and len(self.current_annotation) > 2:
        #     if self.main_window:
        #         self.main_window.finish_polygon()
        pass

    def finish_polygon(self):
        """Finish the current polygon annotation."""
        if self.drawing_polygon and len(self.current_annotation) > 2:
            self.drawing_polygon = False
            if self.main_window:
                self.main_window.finish_polygon() # Call main window method


    def start_polygon_edit(self, pos):
        """Find and select a polygon for editing based on click position."""
        for class_name, annotations in self.annotations.items():
            # Make sure class is visible before allowing edit
            if not self.main_window.is_class_visible(class_name):
                continue
            for annotation in reversed(annotations): # Check topmost annotations first
                if "segmentation" in annotation:
                     seg = annotation["segmentation"]
                     # Handle potential nested list from COCO import
                     if seg and isinstance(seg[0], list): seg = seg[0]
                     if len(seg) >= 6:
                         points = [QPoint(int(x), int(y)) for x, y in zip(seg[0::2], seg[1::2])]
                         if self.point_in_polygon(pos, points):
                             self.editing_polygon = annotation
                             self.current_tool = None # Exit drawing tools
                             self.main_window.disable_tools()
                             self.main_window.reset_tool_buttons()
                             self.main_window.enter_edit_mode(annotation) # Notify main window
                             return annotation
        return None

    def handle_editing_click(self, pos, event):
        """Handle clicks during polygon editing."""
        if not self.editing_polygon or "segmentation" not in self.editing_polygon: return

        segmentation = self.editing_polygon["segmentation"]
        # Handle potential nested list from COCO import
        if segmentation and isinstance(segmentation[0], list): segmentation = segmentation[0]

        points = [QPoint(int(x), int(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
        handle_radius = max(2.0, 5.0 / self.zoom_factor)

        # Check for click on existing point
        for i, point in enumerate(points):
            if self.distance(pos, point) < handle_radius:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier: # Use enum
                    # Delete point, ensure at least 3 points remain
                    if len(points) > 3:
                        del segmentation[i*2:i*2+2]
                        # Update bbox/area if they exist in the annotation
                        if 'bbox' in self.editing_polygon:
                            self.editing_polygon['bbox'] = self.main_window.utils.calculate_bbox(segmentation)
                        if 'area' in self.editing_polygon:
                             self.editing_polygon['area'] = self.main_window.utils.calculate_area(self.editing_polygon)
                        self.main_window.update_annotation_list() # Update list display
                        self.main_window.save_current_annotations() # Save change
                    else:
                        QMessageBox.warning(self.main_window, "Cannot Delete", "Polygon must have at least 3 points.")
                else:
                    # Start moving point
                    self.editing_point_index = i
                self.update()
                return

        # Check for click on line segment to add new point
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            if self.point_on_line(pos, p1, p2, tolerance=3.0/self.zoom_factor): # Add tolerance
                # Insert new point into the flat list
                insert_index = (i + 1) * 2
                segmentation.insert(insert_index, pos[1]) # Insert Y first
                segmentation.insert(insert_index, pos[0]) # Insert X
                self.editing_point_index = i + 1 # Start dragging the newly added point
                self.update()
                return

    def handle_editing_move(self, pos):
        """Handle mouse movement during polygon editing."""
        if not self.editing_polygon or "segmentation" not in self.editing_polygon: return

        segmentation = self.editing_polygon["segmentation"]
        # Handle potential nested list
        if segmentation and isinstance(segmentation[0], list): segmentation = segmentation[0]

        points = [QPoint(int(x), int(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
        handle_radius = max(2.0, 5.0 / self.zoom_factor)
        self.hover_point_index = None

        # Check hover over points
        for i, point in enumerate(points):
            if self.distance(pos, point) < handle_radius:
                self.hover_point_index = i
                break

        # Update position if dragging a point
        if self.editing_point_index is not None and self.editing_point_index < len(points):
            segmentation[self.editing_point_index*2] = pos[0]
            segmentation[self.editing_point_index*2+1] = pos[1]


    def exit_editing_mode(self):
        """Cleans up state when exiting polygon edit mode."""
        # Save changes made during editing before clearing state
        if self.editing_polygon:
             if 'bbox' in self.editing_polygon:
                 self.editing_polygon['bbox'] = self.main_window.utils.calculate_bbox(self.editing_polygon['segmentation'])
             if 'area' in self.editing_polygon:
                  self.editing_polygon['area'] = self.main_window.utils.calculate_area(self.editing_polygon)
             self.main_window.update_annotation_list() # Update list display
             self.main_window.save_current_annotations() # Save changes

        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.update()
        # Notify main window if needed (already done via enter_edit_mode callback?)


    @staticmethod
    def point_in_polygon(point_coords, polygon_points):
        """Check if a point (tuple) is inside a polygon (list of QPoint)."""
        # Convert polygon QPoints to tuples for Shapely
        poly_tuples = [(p.x(), p.y()) for p in polygon_points]
        if len(poly_tuples) < 3:
            return False
        try:
            point_geom = Point(point_coords)
            polygon_geom = Polygon(poly_tuples)
            return polygon_geom.contains(point_geom)
        except Exception as e: # Handle potential Shapely errors
             print(f"Error in point_in_polygon check: {e}")
             return False # Default to false on error

    @staticmethod
    def point_to_tuple(point):
        """Convert QPoint or tuple to tuple."""
        if isinstance(point, QPointF): # Handle QPointF too
            return (point.x(), point.y())
        elif isinstance(point, QPoint):
            return (point.x(), point.y())
        return point # Assume it's already a tuple/list

    @staticmethod
    def distance(p1, p2):
        """Calculate distance between two points (tuples or QPoints)."""
        p1_t = ImageLabel.point_to_tuple(p1)
        p2_t = ImageLabel.point_to_tuple(p2)
        try:
            return ((p1_t[0] - p2_t[0])**2 + (p1_t[1] - p2_t[1])**2)**0.5
        except (TypeError, IndexError):
            return float('inf') # Return infinity if points are invalid

    @staticmethod
    def point_on_line(p, start, end, tolerance=0.5):
        """Check if a point is on a line segment within a tolerance."""
        p_t = ImageLabel.point_to_tuple(p)
        start_t = ImageLabel.point_to_tuple(start)
        end_t = ImageLabel.point_to_tuple(end)
        # Check if point is within bounding box of segment (with tolerance)
        if not (min(start_t[0], end_t[0]) - tolerance <= p_t[0] <= max(start_t[0], end_t[0]) + tolerance and
                min(start_t[1], end_t[1]) - tolerance <= p_t[1] <= max(start_t[1], end_t[1]) + tolerance):
            return False
        # Check distance from point to the infinite line defined by start/end
        try:
            # Using cross-product method / distance from point to line
            dxc = p_t[0] - start_t[0]
            dyc = p_t[1] - start_t[1]
            dxl = end_t[0] - start_t[0]
            dyl = end_t[1] - start_t[1]
            cross = dxc * dyl - dyc * dxl
            if abs(dxl) > 1e-6 or abs(dyl) > 1e-6: # Avoid division by zero for zero-length segment
                dist_sq = cross**2 / (dyl**2 + dxl**2)
                return dist_sq <= tolerance**2
            else: # Zero-length segment, check distance to the single point
                 return ImageLabel.distance(p_t, start_t) <= tolerance
        except (ZeroDivisionError, TypeError, IndexError):
             return False
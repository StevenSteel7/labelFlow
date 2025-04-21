# stack_to_slices.py (Migrated to PyQt6)
import os
import numpy as np
import traceback # For detailed error logging

# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QMessageBox, QComboBox, QGridLayout, QWidget,
                             QProgressDialog, QApplication, QDialogButtonBox) # Added QDialogButtonBox
from PyQt6.QtCore import Qt

# External Libraries
from tifffile import TiffFile
from czifile import CziFile
from PIL import Image # Requires Pillow

# --- Dimension Dialog (Copied - Needs PyQt6 adjustment) ---
class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        self.shape = shape
        self.initUI(file_name)

    def initUI(self, file_name):
        layout = QVBoxLayout(self)

        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)

        layout.addWidget(QLabel(f"Image shape: {self.shape}"))
        layout.addWidget(QLabel("Assign dimensions (H=Height, W=Width):"))

        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        dimensions = ['T', 'Z', 'C', 'S', 'H', 'W'] # Available dimensions
        # Basic suggestion logic (can be improved)
        suggested_dims = [''] * len(self.shape)
        if len(self.shape) >= 2: suggested_dims[-1] = 'W'; suggested_dims[-2] = 'H'
        if len(self.shape) >= 3: suggested_dims[-3] = 'C' if self.shape[-3] <= 5 else 'Z'
        if len(self.shape) >= 4: suggested_dims[-4] = 'T'

        for i, dim_size in enumerate(self.shape):
            dim_layout.addWidget(QLabel(f"Dim {i} (size {dim_size}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            # Set suggested dimension if valid
            if suggested_dims[i] in dimensions:
                 index = combo.findText(suggested_dims[i])
                 if index >= 0: combo.setCurrentIndex(index)
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)

        # Use standard buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel) # Use enum
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]


# --- Main StackToSlices Dialog ---
class StackToSlicesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stack to Slices Converter")
        self.setGeometry(100, 100, 450, 250) # Adjusted size
        # Use PyQt6 enums
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.dimensions = None
        self.file_name = None # Initialize file_name
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- File Selection ---
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No stack file selected")
        self.file_label.setWordWrap(True)
        select_button = QPushButton("Select Stack File...")
        select_button.setToolTip("Select a multi-page TIFF or CZI file")
        select_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label, 1) # Allow label to stretch
        file_layout.addWidget(select_button)
        layout.addLayout(file_layout)

        # --- Convert Button (enabled after dimension assignment) ---
        self.convert_button = QPushButton("Convert to Slices")
        self.convert_button.clicked.connect(self.convert_to_slices)
        self.convert_button.setEnabled(False)
        self.convert_button.setToolTip("Assign dimensions first using the 'Select Stack File' button")
        layout.addWidget(self.convert_button)

        self.setLayout(layout)

    def select_file(self):
        """Opens file dialog and triggers dimension assignment."""
        selected_file, _ = QFileDialog.getOpenFileName(
            self, "Select Stack File", "", "Image Stacks (*.tif *.tiff *.czi);;All Files (*)"
        )
        if selected_file:
            self.file_name = selected_file
            self.file_label.setText(f"Selected: ...{os.sep}{os.path.basename(self.file_name)}")
            self.file_label.setToolTip(self.file_name)
            QApplication.processEvents()
            self.process_file_for_dimensions() # Renamed for clarity
        else:
            self.file_name = None
            self.dimensions = None
            self.file_label.setText("No stack file selected")
            self.file_label.setToolTip("")
            self.convert_button.setEnabled(False)

    def process_file_for_dimensions(self):
        """Reads stack shape and prompts user for dimension assignment."""
        if not self.file_name: return
        image_array = None
        shape = None
        try:
            if self.file_name.lower().endswith(('.tif', '.tiff')):
                with TiffFile(self.file_name) as tif:
                    # Load only the shape first if possible, or first page shape
                    # shape = tif.series[0].shape # More robust way to get shape
                    # For simplicity, load the array to get shape
                    image_array = tif.asarray()
                    shape = image_array.shape
            elif self.file_name.lower().endswith('.czi'):
                with CziFile(self.file_name) as czi:
                    image_array = czi.asarray() # CZI usually needs reading
                    shape = image_array.shape
            else:
                 QMessageBox.warning(self, "Unsupported Format", "Only TIFF and CZI stack files are supported.")
                 return

            if shape and len(shape) > 2: # Only ask for dimensions if multi-dimensional
                 self.get_dimensions(shape)
            elif shape and len(shape) <= 2:
                 QMessageBox.information(self, "Not a Stack", "The selected file appears to be a 2D image, not a stack.")
                 self.dimensions = None
                 self.convert_button.setEnabled(False)
            else:
                 raise ValueError("Could not determine image shape.")

        except Exception as e:
            QMessageBox.critical(self, "Error Reading File", f"Could not read file shape or data:\n{e}")
            traceback.print_exc()
            self.dimensions = None
            self.convert_button.setEnabled(False)


    def get_dimensions(self, shape):
        """Opens the dimension assignment dialog."""
        dialog = DimensionDialog(shape, os.path.basename(self.file_name), self)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal) # Use enum
        result = dialog.exec() # Use exec()
        if result == QDialog.DialogCode.Accepted: # Use enum
            assigned_dims = dialog.get_dimensions()
            if 'H' in assigned_dims and 'W' in assigned_dims:
                self.dimensions = assigned_dims
                print(f"Assigned dimensions: {self.dimensions}")
                self.convert_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Invalid Dimensions", "Assigned dimensions MUST include 'H' (Height) and 'W' (Width).")
                self.dimensions = None
                self.convert_button.setEnabled(False)
        else:
            print("Dimension assignment cancelled.")
            self.dimensions = None
            self.convert_button.setEnabled(False)
        QApplication.processEvents()

    def convert_to_slices(self):
        """Initiates the slice saving process."""
        if not self.file_name or not self.dimensions:
            QMessageBox.warning(self, "Invalid Input", "Please select a file and successfully assign dimensions first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Slices")
        if not output_dir:
            return

        image_array = None
        try:
             # Reload the array data
             print(f"Reloading data from {self.file_name}...")
             if self.file_name.lower().endswith(('.tif', '.tiff')):
                 with TiffFile(self.file_name) as tif:
                      image_array = tif.asarray()
             elif self.file_name.lower().endswith('.czi'):
                 with CziFile(self.file_name) as czi:
                      image_array = czi.asarray()
             if image_array is None: raise ValueError("Failed to reload image array.")
             print(f"Data reloaded, shape: {image_array.shape}")

             self.save_slices(image_array, output_dir)

        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"An error occurred during conversion:\n{e}")
            traceback.print_exc()


    def save_slices(self, image_array, output_dir):
        """Iterates through non-HW dimensions and saves each HW slice."""
        base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        print(f"Saving slices for {base_name} with dimensions {self.dimensions}...")

        try:
            h_index = self.dimensions.index('H')
            w_index = self.dimensions.index('W')
        except ValueError:
             QMessageBox.critical(self, "Dimension Error", "Internal error: H or W not found in assigned dimensions.")
             return

        # Indices and shapes of non-HW dimensions
        non_hw_indices = [i for i, dim in enumerate(self.dimensions) if dim not in ['H', 'W']]
        non_hw_shapes = [image_array.shape[i] for i in non_hw_indices]

        total_slices_to_save = int(np.prod(non_hw_shapes)) if non_hw_shapes else 1
        if total_slices_to_save == 0:
             QMessageBox.information(self, "No Slices", "Calculated zero slices to save based on dimensions.")
             return

        print(f"Total slices to save: {total_slices_to_save}")

        progress = QProgressDialog("Saving slices...", "Cancel", 0, total_slices_to_save, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal) # Use enum
        progress.setWindowTitle("Saving Progress")
        progress.setMinimumDuration(0) # Show immediately
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        saved_count = 0
        try:
            # Iterate using np.ndindex over the shapes of non-HW dimensions
            for slice_idx_tuple in np.ndindex(tuple(non_hw_shapes)):
                if progress.wasCanceled():
                    break

                # Construct the multi-dimensional index to access the HW slice
                full_idx = [slice(None)] * image_array.ndim # Start with all slices selected
                for i, original_dim_index in enumerate(non_hw_indices):
                    full_idx[original_dim_index] = slice_idx_tuple[i] # Set specific index for non-HW dims

                # Extract the 2D (or potentially 3D if C exists) slice
                slice_array = image_array[tuple(full_idx)]

                # If Channel 'C' exists and is not H or W, handle it
                if 'C' in self.dimensions:
                    c_index_relative = -1
                    c_dim_original = -1
                    # Find original index of C relative to *all* dimensions
                    try: c_dim_original = self.dimensions.index('C')
                    except ValueError: pass

                    if c_dim_original != -1 and c_dim_original != h_index and c_dim_original != w_index:
                         # Determine C index relative to the *extracted* slice_array dimensions
                         original_indices_in_slice = sorted([h_index, w_index] + [idx for idx in non_hw_indices if idx != c_dim_original])
                         current_indices_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices_in_slice)}

                         # Check if C dimension was sliced out or still exists
                         if c_dim_original not in original_indices_in_slice:
                              # C dimension was iterated over by ndindex, handle below
                              pass
                         else:
                              # C dimension is still present in slice_array, need to handle it
                              # This case implies saving multi-channel images, which might need adjustment
                              # For simplicity, let's assume we save each channel as grayscale, or save as RGB if 3 channels
                              print(f"Warning: Slice still contains Channel dimension (shape: {slice_array.shape}). Handling C dimension...")
                              # Example: Save each channel separately or try to form RGB
                              # This part needs refinement based on desired output for multi-channel slices
                              pass # Placeholder - current logic might save multi-channel slices directly if PIL supports it

                # Squeeze out dimensions that were iterated over (should leave H, W, maybe C)
                slice_array = slice_array.squeeze()
                if slice_array.ndim == 0: continue # Skip if squeezed to single value

                # Determine PIL mode based on dtype and shape
                mode = None
                if slice_array.ndim == 2: # Grayscale
                    if slice_array.dtype == np.uint16: mode = 'I;16'
                    elif slice_array.dtype == np.uint8: mode = 'L'
                    else: # Attempt conversion to uint16 for other numeric types
                         print(f"Warning: Converting slice dtype {slice_array.dtype} to uint16.")
                         slice_min, slice_max = np.min(slice_array), np.max(slice_array)
                         if slice_max > slice_min:
                              slice_array = ((slice_array - slice_min) / (slice_max - slice_min) * 65535).astype(np.uint16)
                         else:
                              slice_array = np.zeros_like(slice_array, dtype=np.uint16)
                         mode = 'I;16'
                elif slice_array.ndim == 3: # Assume Color (RGB or RGBA?)
                     if slice_array.shape[-1] == 3: mode = 'RGB'
                     elif slice_array.shape[-1] == 4: mode = 'RGBA'
                     # Ensure uint8 for standard color modes
                     if slice_array.dtype != np.uint8:
                          print(f"Warning: Converting color slice dtype {slice_array.dtype} to uint8.")
                          slice_array = np.clip(slice_array, 0, 255).astype(np.uint8) # Basic clip/cast

                if mode is None:
                     print(f"Warning: Skipping slice - could not determine save mode for shape {slice_array.shape} and dtype {slice_array.dtype}")
                     continue

                # Construct filename including non-HW dimension indices
                slice_name_parts = [f'{self.dimensions[non_hw_indices[i]]}{slice_idx_tuple[i]}' for i in range(len(slice_idx_tuple))]
                slice_filename = f"{base_name}_{'_'.join(slice_name_parts)}.png" # Save as PNG
                output_path = os.path.join(output_dir, slice_filename)

                try:
                    img = Image.fromarray(slice_array, mode=mode)
                    img.save(output_path)
                    saved_count += 1
                except Exception as save_err:
                     print(f"Error saving slice {output_path}: {save_err}")

                progress.setValue(saved_count)
                QApplication.processEvents()

            progress.setValue(total_slices_to_save) # Ensure progress reaches 100%

            if progress.wasCanceled():
                QMessageBox.warning(self, "Conversion Interrupted", f"Conversion cancelled. {saved_count} slices were saved.")
            else:
                QMessageBox.information(self, "Conversion Complete", f"{saved_count} slices have been saved to:\n{output_dir}")

        except Exception as e:
             QMessageBox.critical(self, "Error", f"An error occurred during slice saving:\n{e}")
             traceback.print_exc()
        finally:
            if progress: progress.close()


    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            if self.rect().isValid():
                 self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()

# Function to show the dialog
def show_stack_to_slices(parent):
    """Creates and shows the Stack to Slices dialog."""
    dialog = StackToSlicesDialog(parent)
    dialog.show_centered(parent)
    return dialog # Return instance if needed
# image_patcher.py (Migrated to PyQt6)
import os
import numpy as np
import traceback # For detailed error logging

# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QProgressBar, QMessageBox, QListWidget, QDialogButtonBox,
                             QGridLayout, QComboBox, QApplication, QScrollArea, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# External libraries
from tifffile import TiffFile, imwrite
from PIL import Image # PIL uses Pillow fork usually

# --- Dimension Dialog (Copied from annotator_window - needs PyQt6 adjustment) ---
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
        suggested_dims = list(' ' * len(self.shape)) # Placeholder for suggestions
        # Simple suggestion logic (can be improved)
        if len(self.shape) >= 2:
            suggested_dims[-1] = 'W'
            suggested_dims[-2] = 'H'
        if len(self.shape) >= 3:
            suggested_dims[-3] = 'C' if self.shape[-3] <= 5 else 'Z' # Guess Channel or Z
        if len(self.shape) >= 4:
            suggested_dims[-4] = 'T' # Guess Time

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
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]

# --- Patching Thread ---
class PatchingThread(QThread):
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str) # Signal for logging status/warnings
    error = pyqtSignal(str)
    finished = pyqtSignal()
    dimension_required = pyqtSignal(tuple, str) # Emit shape and filename

    def __init__(self, input_files, output_dir, patch_size, overlap, dimensions):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.patch_size = patch_size # Should be (height, width) tuple
        self.overlap = overlap    # Should be (y_overlap, x_overlap) tuple
        self.dimensions = dimensions # Dictionary: {file_path: ['T', 'Z', 'H', 'W'...]}
        self._dimension_response = None # To store response from main thread
        self._wait_for_dimension = False # Flag to pause execution
        self.is_cancelled = False

    def run(self):
        try:
            total_files = len(self.input_files)
            for i, file_path in enumerate(self.input_files):
                if self.is_cancelled:
                    self.log_message.emit("Patching cancelled.")
                    break
                self.log_message.emit(f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
                self.patch_image(file_path)
                self.progress.emit(int((i + 1) / total_files * 100))

            if not self.is_cancelled:
                 self.log_message.emit("Patching finished successfully.")
                 self.finished.emit()
            else:
                 # Emit finished anyway so UI resets, but maybe indicate cancellation?
                 # Or add a different signal for cancellation.
                 self.finished.emit()

        except Exception as e:
            error_msg = f"Error during patching: {str(e)}"
            self.log_message.emit(error_msg)
            self.error.emit(error_msg)
            traceback.print_exc()

    def request_dimension(self, shape, file_name):
        """Signals main thread that dimensions are needed and waits."""
        self._dimension_response = None
        self._wait_for_dimension = True
        self.dimension_required.emit(shape, file_name)
        # Basic wait loop - not ideal, but simple for this structure
        while self._wait_for_dimension and not self.is_cancelled:
            self.msleep(100) # Sleep briefly to yield control
        return self._dimension_response # Return the dimensions received or None

    def set_dimension_response(self, dimensions):
        """Called by the main thread to provide dimensions."""
        self._dimension_response = dimensions
        self._wait_for_dimension = False # Release the wait loop

    def patch_image(self, file_path):
        """Processes a single image file (multi-dimensional or 2D)."""
        file_name = os.path.basename(file_path)
        file_name_without_ext, file_extension = os.path.splitext(file_name)

        try:
            if file_extension.lower() in ['.tif', '.tiff']:
                with TiffFile(file_path) as tif:
                    # Check if it's an OME-TIFF with metadata first
                    ome_metadata = None
                    if tif.is_ome:
                         try: ome_metadata = tif.ome_metadata # Access OME metadata if present
                         except Exception as ome_err: print(f"Note: Error reading OME metadata for {file_name}: {ome_err}")

                    images = tif.asarray() # Load the whole array
                    self.log_message.emit(f"Loaded TIFF: {file_name}, Shape: {images.shape}, Dtype: {images.dtype}")

                    if images.ndim <= 2: # Simple 2D TIFF
                         self.save_patches(images, file_name_without_ext, file_extension)
                    else: # Multi-dimensional TIFF
                         # Try getting dimensions from OME metadata if available
                         dims_from_ome = None
                         if ome_metadata and isinstance(ome_metadata, str): # Basic check if it's a string
                             try:
                                  # Basic parsing, might need xml.etree.ElementTree for complex cases
                                  if 'DimensionOrder="' in ome_metadata:
                                       order_str = ome_metadata.split('DimensionOrder="')[1].split('"')[0]
                                       if len(order_str) == images.ndim:
                                            dims_from_ome = list(order_str.upper()) # Use order directly
                                            print(f"Using DimensionOrder from OME metadata: {dims_from_ome}")
                             except Exception as parse_err:
                                  print(f"Note: Error parsing OME DimensionOrder for {file_name}: {parse_err}")

                         # Use dimensions from OME > stored > ask user
                         current_dims = dims_from_ome or self.dimensions.get(file_path)
                         if not current_dims:
                              self.log_message.emit(f"Dimensions needed for {file_name} (Shape: {images.shape})")
                              current_dims = self.request_dimension(images.shape, file_name)
                              if current_dims:
                                   self.dimensions[file_path] = current_dims # Store for potential future use
                              else:
                                   self.log_message.emit(f"Dimension assignment skipped or failed for {file_name}. Skipping file.")
                                   return # Skip this file if dimensions not provided

                         if 'H' not in current_dims or 'W' not in current_dims:
                              raise ValueError(f"Dimensions assigned for {file_name} ({current_dims}) must include 'H' and 'W'.")

                         h_index = current_dims.index('H')
                         w_index = current_dims.index('W')

                         # Create iterators for non-HW dimensions
                         non_hw_indices = [i for i, dim in enumerate(current_dims) if dim not in ['H', 'W']]
                         non_hw_shapes = [images.shape[i] for i in non_hw_indices]

                         total_slices_to_patch = np.prod(non_hw_shapes) if non_hw_shapes else 1
                         slice_patch_count = 0

                         self.log_message.emit(f"Patching {total_slices_to_patch} slice(s) from {file_name}...")

                         # Iterate through all non-spatial slices
                         for idx_tuple in np.ndindex(tuple(non_hw_shapes)):
                              if self.is_cancelled: return # Check cancellation within loop

                              # Construct the multi-dimensional index for slicing
                              slice_access_index = [slice(None)] * images.ndim
                              for i, non_hw_idx_pos in enumerate(non_hw_indices):
                                   slice_access_index[non_hw_idx_pos] = idx_tuple[i]

                              # Extract the 2D slice corresponding to H and W
                              image_slice = images[tuple(slice_access_index)]

                              # Construct a descriptive name for the slice
                              slice_desc = '_'.join([f'{current_dims[non_hw_indices[i]]}{idx_tuple[i]}' for i in range(len(idx_tuple))])
                              base_name_for_patches = f"{file_name_without_ext}_{slice_desc}" if slice_desc else file_name_without_ext

                              self.save_patches(image_slice, base_name_for_patches, file_extension)
                              slice_patch_count += 1
                              # Optional: emit progress based on slice patching if needed
                              # self.log_message.emit(f"Patched slice {slice_patch_count}/{total_slices_to_patch}")


            else: # Handle standard image formats (PNG, JPG, etc.)
                 with Image.open(file_path) as img:
                      image = np.array(img)
                      self.log_message.emit(f"Loaded Image: {file_name}, Shape: {image.shape}, Dtype: {image.dtype}")
                      if image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [3, 4]): # Allow 2D or 3D (RGB/RGBA)
                           self.save_patches(image, file_name_without_ext, file_extension)
                      else:
                           self.log_message.emit(f"Warning: Skipping non-standard image {file_name} with shape {image.shape}")

        except Exception as e:
             self.log_message.emit(f"ERROR processing {file_name}: {e}")
             traceback.print_exc() # Log detailed error
             # Optionally re-raise or emit error signal if needed
             # self.error.emit(f"Error processing {file_name}: {e}")


    def save_patches(self, image_2d, base_name, extension):
        """Saves patches from a single 2D image slice."""
        if image_2d.ndim != 2 and not (image_2d.ndim == 3 and image_2d.shape[2] in [3, 4]):
             self.log_message.emit(f"Warning: Skipping patch saving for '{base_name}'. Expected 2D image, got shape {image_2d.shape}")
             return

        h, w = image_2d.shape[:2]
        patch_h, patch_w = self.patch_size
        overlap_y, overlap_x = self.overlap # Correct order: Y first, then X

        # Calculate step size based on patch size and overlap
        step_y = patch_h - overlap_y
        step_x = patch_w - overlap_x

        # Ensure step size is positive
        if step_y <= 0 or step_x <= 0:
            raise ValueError(f"Overlap ({overlap_x}, {overlap_y}) must be smaller than Patch Size ({patch_w}, {patch_h})")

        patch_count = 0
        # Iterate using steps, ensuring we don't go out of bounds
        for i in range(0, h - patch_h + 1, step_y): # Stop if remaining height < patch_h
            for j in range(0, w - patch_w + 1, step_x): # Stop if remaining width < patch_w
                patch = image_2d[i : i + patch_h, j : j + patch_w]
                patch_name = f"{base_name}_patch_{i}_{j}{extension}" # y-coord first in name
                output_path = os.path.join(self.output_dir, patch_name)

                try:
                    if extension.lower() in ['.tif', '.tiff']:
                        # Preserve dtype for TIFF
                        imwrite(output_path, patch)
                    else:
                        # Convert to PIL Image for saving standard formats
                        # Handle different numpy dtypes appropriately for PIL
                        if patch.dtype == np.uint16:
                            # PIL handles 'I;16' for 16-bit grayscale
                             mode = 'I;16' if patch.ndim == 2 else None # Need mode for RGB 16bit?
                             if patch.ndim == 3:
                                 print(f"Warning: Saving 16-bit color patch {patch_name} might lose precision.")
                                 # Convert to 8-bit RGB for standard formats
                                 patch_8bit = (np.clip(patch / 256, 0, 255)).astype(np.uint8)
                                 Image.fromarray(patch_8bit, mode='RGB').save(output_path)
                             elif mode:
                                 Image.fromarray(patch, mode=mode).save(output_path)
                             else:
                                 print(f"Warning: Skipping patch {patch_name}, unsupported 16-bit format for PIL.")
                        elif patch.ndim == 3 and patch.shape[2] == 4: # RGBA
                            Image.fromarray(patch, mode='RGBA').save(output_path)
                        elif patch.ndim == 3 and patch.shape[2] == 3: # RGB
                            Image.fromarray(patch, mode='RGB').save(output_path)
                        elif patch.ndim == 2: # Grayscale (assume uint8 or convert)
                            if patch.dtype != np.uint8: # Convert other grayscale types to uint8
                                patch = (np.clip(patch, 0, 255)).astype(np.uint8)
                            Image.fromarray(patch, mode='L').save(output_path)
                        else:
                             print(f"Warning: Skipping patch {patch_name}, unsupported shape/channels for standard image saving.")
                    patch_count += 1
                except Exception as save_err:
                    self.log_message.emit(f"Error saving patch {output_path}: {save_err}")

        # Log leftovers calculation (no patch saving for leftovers)
        patches_x_full = (w - patch_w) // step_x + 1 if w >= patch_w else 0
        patches_y_full = (h - patch_h) // step_y + 1 if h >= patch_h else 0
        last_x_start = (patches_x_full - 1) * step_x if patches_x_full > 0 else 0
        last_y_start = (patches_y_full - 1) * step_y if patches_y_full > 0 else 0
        covered_w = last_x_start + patch_w
        covered_h = last_y_start + patch_h
        leftover_x = w - covered_w
        leftover_y = h - covered_h

        self.log_message.emit(f"Saved {patch_count} patches for '{base_name}'. Leftover Pixels: X={leftover_x}, Y={leftover_y}")

    def cancel(self):
        self.is_cancelled = True
        if self._wait_for_dimension: # If waiting for dimensions, release the wait
             self._wait_for_dimension = False
        self.log_message.emit("Cancellation requested.")


# --- Main Patcher Tool Dialog ---
class ImagePatcherTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal) # Use enum
        self.dimensions = {} # Store assigned dimensions {file_path: list_of_dims}
        self.input_files = []
        self.output_dir = ""
        self.patching_thread = None # Thread reference
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Input/Output ---
        io_group = QGroupBox("Input / Output")
        io_layout = QGridLayout()
        io_layout.setColumnStretch(1, 1)
        io_layout.setSpacing(8)

        self.input_label = QLabel("No files selected")
        input_button = QPushButton("...")
        input_button.setFixedSize(30, 25)
        input_button.setToolTip("Select Input Image Files")
        input_button.clicked.connect(self.select_input_files)
        io_layout.addWidget(QLabel("Input Files:"), 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        io_layout.addWidget(self.input_label, 0, 1)
        io_layout.addWidget(input_button, 0, 2)

        self.output_label = QLabel("Not selected")
        output_button = QPushButton("...")
        output_button.setFixedSize(30, 25)
        output_button.setToolTip("Select Output Directory")
        output_button.clicked.connect(self.select_output_directory)
        io_layout.addWidget(QLabel("Output Dir:"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
        io_layout.addWidget(self.output_label, 1, 1)
        io_layout.addWidget(output_button, 1, 2)

        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        # --- Patch Settings ---
        settings_group = QGroupBox("Patch Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(8)

        settings_layout.addWidget(QLabel("Patch Width:"), 0, 0)
        self.patch_w = QSpinBox()
        self.patch_w.setRange(16, 10000) # Min patch size 16
        self.patch_w.setValue(256)
        settings_layout.addWidget(self.patch_w, 0, 1)

        settings_layout.addWidget(QLabel("Patch Height:"), 1, 0)
        self.patch_h = QSpinBox()
        self.patch_h.setRange(16, 10000)
        self.patch_h.setValue(256)
        settings_layout.addWidget(self.patch_h, 1, 1)

        settings_layout.addWidget(QLabel("Overlap X:"), 0, 2)
        self.overlap_x = QSpinBox()
        self.overlap_x.setRange(0, 9999) # Overlap must be less than patch size
        self.overlap_x.setValue(0)
        settings_layout.addWidget(self.overlap_x, 0, 3)

        settings_layout.addWidget(QLabel("Overlap Y:"), 1, 2)
        self.overlap_y = QSpinBox()
        self.overlap_y.setRange(0, 9999)
        self.overlap_y.setValue(0)
        settings_layout.addWidget(self.overlap_y, 1, 3)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Patch Info Area ---
        info_group = QGroupBox("Estimated Patch Info (for selected files)")
        info_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(150) # Set fixed height

        self.patch_info_label = QLabel("Select input files and settings to see estimates.")
        self.patch_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft) # Use enum
        self.patch_info_label.setWordWrap(True)
        scroll_area.setWidget(self.patch_info_label)
        info_layout.addWidget(scroll_area)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- Action Buttons & Progress ---
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Patching")
        self.start_button.clicked.connect(self.start_patching)
        self.stop_button = QPushButton("Cancel")
        self.stop_button.clicked.connect(self.cancel_patching)
        self.stop_button.setEnabled(False)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)


        self.setWindowTitle('Image Patcher Tool')
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)

        # Connect value changed signals to update info and validate overlap
        self.patch_w.valueChanged.connect(self.validate_overlap_and_update_info)
        self.patch_h.valueChanged.connect(self.validate_overlap_and_update_info)
        self.overlap_x.valueChanged.connect(self.validate_overlap_and_update_info)
        self.overlap_y.valueChanged.connect(self.validate_overlap_and_update_info)

    def select_input_files(self):
        """Select multiple input image files."""
        # QFileDialog.getOpenFileNames returns tuple (fileNames, selectedFilter)
        selected_files, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Files", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if selected_files:
            self.input_files = selected_files
            self.input_label.setText(f"{len(self.input_files)} files selected")
            self.input_label.setToolTip("\n".join(self.input_files)) # Show list in tooltip
            QApplication.processEvents()
            self.dimensions.clear() # Clear old dimensions when new files are selected
            self.process_multi_dim_files() # Check dimensions immediately
            self.update_patch_info() # Update info after files selected
        else:
            self.input_files = []
            self.input_label.setText("No files selected")
            self.input_label.setToolTip("")
            self.update_patch_info()

    def process_multi_dim_files(self):
        """Checks dimensions for TIFF files that might be multi-dimensional."""
        multi_dim_files_to_ask = []
        for file_path in self.input_files:
            if file_path.lower().endswith(('.tif', '.tiff')):
                 try:
                     with TiffFile(file_path) as tif:
                          # Only ask if ndim > 2 and not explicitly assigned yet
                          if tif.asarray().ndim > 2 and file_path not in self.dimensions:
                               multi_dim_files_to_ask.append(file_path)
                 except Exception as e:
                      self.log_status(f"Warning: Could not read {os.path.basename(file_path)} to check dimensions: {e}")

        # Ask for dimensions sequentially
        for file_path in multi_dim_files_to_ask:
             self.check_tiff_dimensions(file_path) # This will prompt the user


    def check_tiff_dimensions(self, file_path):
        """Opens the dimension dialog for a specific multi-dimensional TIFF file."""
        try:
            with TiffFile(file_path) as tif:
                images = tif.asarray()
                if images.ndim > 2:
                    file_name = os.path.basename(file_path)
                    dialog = DimensionDialog(images.shape, file_name, self)
                    dialog.setWindowModality(Qt.WindowModality.ApplicationModal) # Use enum
                    result = dialog.exec() # Use exec()
                    if result == QDialog.DialogCode.Accepted: # Use enum
                        dims = dialog.get_dimensions()
                        if 'H' in dims and 'W' in dims:
                            self.dimensions[file_path] = dims
                            self.log_status(f"Assigned dimensions for {file_name}: {dims}")
                        else:
                            QMessageBox.warning(self, "Invalid Dimensions", f"Dimensions for {file_name} MUST include 'H' (Height) and 'W' (Width).")
                            self.dimensions.pop(file_path, None) # Remove invalid entry
                    else:
                         self.log_status(f"Dimension assignment cancelled for {file_name}.")
                         self.dimensions.pop(file_path, None) # Remove if cancelled
                    QApplication.processEvents()
        except Exception as e:
             self.log_status(f"Error checking dimensions for {os.path.basename(file_path)}: {e}")

    def select_output_directory(self):
        """Select the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"...{os.sep}{os.path.basename(directory)}")
            self.output_label.setToolTip(directory)
            self.update_patch_info() # Update info after directory selected

    def validate_overlap_and_update_info(self):
        """Validates overlap values and updates the info label."""
        # Validate Overlap X
        max_overlap_x = self.patch_w.value() - 1
        if self.overlap_x.value() > max_overlap_x:
            self.overlap_x.setValue(max_overlap_x)
            self.log_status(f"Overlap X adjusted to be less than Patch Width ({max_overlap_x})")

        # Validate Overlap Y
        max_overlap_y = self.patch_h.value() - 1
        if self.overlap_y.value() > max_overlap_y:
            self.overlap_y.setValue(max_overlap_y)
            self.log_status(f"Overlap Y adjusted to be less than Patch Height ({max_overlap_y})")

        self.update_patch_info()

    def get_patch_info(self):
        """Calculates estimated patch counts and leftovers for selected files."""
        patch_info_dict = {}
        patch_w = self.patch_w.value()
        patch_h = self.patch_h.value()
        overlap_x = self.overlap_x.value()
        overlap_y = self.overlap_y.value()

        step_x = patch_w - overlap_x
        step_y = patch_h - overlap_y

        if step_x <= 0 or step_y <= 0:
            return None # Invalid settings

        for file_path in self.input_files:
            file_name = os.path.basename(file_path)
            w, h = 0, 0
            dims_str = ""
            try:
                if file_path.lower().endswith(('.tif', '.tiff')):
                    with TiffFile(file_path) as tif:
                        shape = tif.asarray(key=0).shape # Check shape of first page/slice
                        if len(shape) == 2:
                             h, w = shape
                        elif len(shape) > 2:
                             # Use assigned or default H/W dimensions
                             dims = self.dimensions.get(file_path)
                             if dims and 'H' in dims and 'W' in dims:
                                  h_idx, w_idx = dims.index('H'), dims.index('W')
                                  # Get full shape to display dimensions
                                  full_shape = tif.asarray().shape
                                  h, w = full_shape[h_idx], full_shape[w_idx]
                                  dims_str = f" ({'x'.join(map(str, full_shape))}, {','.join(dims)})"
                             else: # Fallback if dims not assigned/valid
                                  h, w = shape[-2], shape[-1] # Assume last two are H, W
                                  dims_str = f" ({'x'.join(map(str, shape))}, Assuming H,W are last dims)"
                        else: raise ValueError("Invalid TIFF dimensions")
                else: # Standard images
                    with Image.open(file_path) as img:
                        w, h = img.size

                # Calculate full patches
                patches_x = (w - patch_w) // step_x + 1 if w >= patch_w else 0
                patches_y = (h - patch_h) // step_y + 1 if h >= patch_h else 0

                # Calculate leftovers
                last_x_start = (patches_x - 1) * step_x if patches_x > 0 else 0
                last_y_start = (patches_y - 1) * step_y if patches_y > 0 else 0
                covered_w = last_x_start + patch_w
                covered_h = last_y_start + patch_h
                leftover_x = w - covered_w
                leftover_y = h - covered_h


                patch_info_dict[file_name] = {
                    'dims': f"{w}x{h}{dims_str}",
                    'patches_x': patches_x,
                    'patches_y': patches_y,
                    'total_patches': patches_x * patches_y,
                    'leftover_x': leftover_x,
                    'leftover_y': leftover_y
                }
            except Exception as e:
                 patch_info_dict[file_name] = {'error': f"Could not read info: {e}"}

        return patch_info_dict

    def update_patch_info(self):
        """Updates the patch information display label."""
        if not self.input_files:
            self.patch_info_label.setText("Select input files and settings to see estimates.")
            return

        patch_info = self.get_patch_info()
        if patch_info is None:
            self.patch_info_label.setText("<font color='red'>Invalid Settings: Overlap >= Patch Size</font>")
            return

        if patch_info:
            info_text = "<html><body>"
            total_files = len(patch_info)
            total_patches_overall = 0
            max_files_to_show = 10 # Limit detailed display for performance
            files_shown = 0

            for file_name, info in patch_info.items():
                 if files_shown < max_files_to_show:
                      info_text += f"<p><b>{file_name}</b> ({info.get('dims', 'N/A')})<br>"
                      if 'error' in info:
                           info_text += f"<font color='red'>Error: {info['error']}</font></p>"
                      else:
                           total_patches = info.get('total_patches', 0)
                           total_patches_overall += total_patches
                           info_text += f"  Est. Patches: {info['patches_x']} x {info['patches_y']} = {total_patches}<br>"
                           info_text += f"  Leftover Pixels: X={info['leftover_x']}, Y={info['leftover_y']}</p>"
                 elif files_shown == max_files_to_show:
                      info_text += "<p><i>... (info for remaining files hidden) ...</i></p>"

                 # Accumulate total patches even if hidden
                 if 'error' not in info:
                      total_patches_overall += info.get('total_patches', 0)

                 files_shown += 1

            info_text += f"<hr><p><b>Total Files:</b> {total_files}<br>"
            info_text += f"<b>Estimated Total Patches:</b> {total_patches_overall}</p>"
            info_text += "</body></html>"
            self.patch_info_label.setText(info_text)
        else:
            self.patch_info_label.setText("Unable to calculate patch information.")

    def start_patching(self):
        """Starts the patching process in a separate thread."""
        if not self.input_files:
            QMessageBox.warning(self, "No Input Files", "Please select input files.")
            return
        if not self.output_dir or not os.path.isdir(self.output_dir):
            QMessageBox.warning(self, "No Output Directory", "Please select a valid output directory.")
            return

        # Check if dimensions are needed for any multi-dim files
        needs_dims = False
        for file_path in self.input_files:
             if file_path.lower().endswith(('.tif', '.tiff')) and file_path not in self.dimensions:
                 try:
                     with TiffFile(file_path) as tif:
                          if tif.asarray().ndim > 2:
                               needs_dims = True
                               break # Only need one file to trigger prompt
                 except Exception as e:
                      self.log_status(f"Warning: Cannot pre-check dims for {os.path.basename(file_path)}: {e}")

        if needs_dims:
             QMessageBox.information(self, "Dimensions Required",
                                      "Some multi-dimensional TIFF files were detected.\n"
                                      "You will be prompted to assign dimensions (like H, W, Z, C, T) for each during the patching process.")
             # User acknowledged, proceed, thread will emit signal when needed

        patch_size = (self.patch_h.value(), self.patch_w.value()) # H, W
        overlap = (self.overlap_y.value(), self.overlap_x.value()) # Y, X

        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Starting...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.patching_thread = PatchingThread(self.input_files, self.output_dir, patch_size, overlap, self.dimensions.copy()) # Pass copy
        self.patching_thread.progress.connect(self.update_progress)
        self.patching_thread.log_message.connect(self.log_status)
        self.patching_thread.error.connect(self.show_error)
        self.patching_thread.finished.connect(self.patching_finished)
        self.patching_thread.dimension_required.connect(self.prompt_for_dimensions) # Connect signal
        self.patching_thread.start()


    def prompt_for_dimensions(self, shape, file_name):
        """Slot to handle the dimension_required signal from the thread."""
        self.log_status(f"Waiting for dimension assignment for {file_name}...")
        dialog = DimensionDialog(shape, file_name, self)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        result = dialog.exec()
        assigned_dims = None
        if result == QDialog.DialogCode.Accepted:
            dims = dialog.get_dimensions()
            if 'H' in dims and 'W' in dims:
                assigned_dims = dims
                self.dimensions[os.path.join(self.input_dir or "", file_name)] = dims # Store persistently for this session
                self.log_status(f"Dimensions received for {file_name}: {dims}")
            else:
                QMessageBox.warning(self, "Invalid Dimensions", f"Dimensions for {file_name} MUST include 'H' (Height) and 'W' (Width). File will be skipped.")
                self.log_status(f"Invalid dimensions for {file_name}. Skipping.")
        else:
             self.log_status(f"Dimension assignment cancelled for {file_name}. Skipping.")

        # Send response back to thread (even if None)
        if self.patching_thread:
             self.patching_thread.set_dimension_response(assigned_dims)


    def cancel_patching(self):
        """Requests the patching thread to stop."""
        if self.patching_thread and self.patching_thread.isRunning():
            self.patching_thread.cancel()
            self.stop_button.setEnabled(False)
            self.start_button.setText("Cancelling...")
            self.status_label.setText("Status: Cancelling...")


    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def log_status(self, message):
        """Updates the status label."""
        self.status_label.setText(f"Status: {message}")
        print(message) # Also print to console for debugging

    def show_error(self, error_message):
        QMessageBox.critical(self, "Patching Error", f"An error occurred:\n{error_message}")
        self.reset_ui_state() # Reset UI on error

    def patching_finished(self):
        """Called when the thread finishes naturally or is cancelled."""
        # Check if cancellation happened (thread sets self.is_cancelled)
        cancelled = self.patching_thread.is_cancelled if self.patching_thread else False
        self.reset_ui_state()
        if not cancelled:
            QMessageBox.information(self, "Patching Complete", "Image patching finished.")
            self.status_label.setText("Status: Finished")
        else:
            QMessageBox.warning(self, "Patching Cancelled", "Image patching was cancelled.")
            self.status_label.setText("Status: Cancelled")
        self.patching_thread = None # Clear thread reference


    def reset_ui_state(self):
        """Resets buttons and status after completion, error, or cancellation."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.start_button.setText("Start Patching")
        # Don't reset progress bar immediately, let user see final state briefly
        # self.progress_bar.setValue(0)
        # self.status_label.setText("Status: Idle")

    def closeEvent(self, event):
        """Handle closing the dialog while patching might be running."""
        if self.patching_thread and self.patching_thread.isRunning():
             reply = QMessageBox.question(self, "Patching Running",
                                          "Patching is in progress. Stop and close?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Use enum
                                          QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes: # Use enum
                 self.cancel_patching()
                 # Allow closing, thread should stop
                 event.accept()
             else:
                 event.ignore()
        else:
             event.accept()

    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            if self.rect().isValid():
                 self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()

# Function to show the dialog
def show_image_patcher(parent=None):
    """Creates and shows the Image Patcher dialog."""
    dialog = ImagePatcherTool(parent)
    dialog.show_centered(parent)
    return dialog # Return instance
# stack_interpolator.py (Migrated to PyQt6)
import os
import numpy as np
import traceback # For detailed error logging

# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                            QLabel, QComboBox, QMessageBox, QProgressDialog, QRadioButton,
                            QButtonGroup, QGroupBox, QDoubleSpinBox, QApplication, QGridLayout) # Added QGridLayout
from PyQt6.QtCore import Qt

# External Libraries
from scipy.interpolate import RegularGridInterpolator
from skimage import io # Requires scikit-image
import tifffile # Requires tifffile

# Custom Exception for user cancellation
class InterruptedError(Exception):
    pass

class StackInterpolator(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stack Interpolator Tool")
        self.setGeometry(100, 100, 600, 550) # Adjusted height
        # Use PyQt6 enums
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Initialize variables
        self.input_path = ""
        self.output_directory = ""

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Input Selection ---
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(5)

        self.dir_radio = QRadioButton("Directory of Image Files (Sorted Numerically)")
        self.stack_radio = QRadioButton("Multi-Page TIFF Stack")
        self.input_type_group = QButtonGroup(self)
        self.input_type_group.addButton(self.dir_radio)
        self.input_type_group.addButton(self.stack_radio)
        self.dir_radio.setChecked(True)
        input_layout.addWidget(self.dir_radio)
        input_layout.addWidget(self.stack_radio)

        file_select_layout = QHBoxLayout()
        self.input_label = QLabel("Input: Not selected")
        self.input_label.setWordWrap(True)
        self.select_input_btn = QPushButton("Select...")
        self.select_input_btn.setToolTip("Select input directory or TIFF stack file")
        self.select_input_btn.clicked.connect(self.select_input)
        file_select_layout.addWidget(self.input_label, 1)
        file_select_layout.addWidget(self.select_input_btn)
        input_layout.addLayout(file_select_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Interpolation Method ---
        method_group = QGroupBox("Interpolation Settings")
        method_layout = QHBoxLayout() # Horizontal layout for label and combo

        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "linear",    # Good balance
            "nearest",   # Fastest, preserves edges, blocky
            # "slinear", # Not directly supported by RegularGridInterpolator
            "cubic",     # Smoother, slower (requires hybrid approach)
            "quintic",   # Smoothest, slowest (requires hybrid approach)
            # "pchip"    # Shape-preserving (requires hybrid approach)
        ])
        self.method_combo.setToolTip("Interpolation algorithm to use.\n"
                                     "'linear' and 'nearest' are faster.\n"
                                     "'cubic' and 'quintic' are smoother but much slower.")
        self.method_combo.setCurrentText("linear") # Default to linear
        method_layout.addWidget(self.method_combo, 1) # Allow combo to stretch
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # --- Dimensions & Spacing ---
        dims_group = QGroupBox("Dimensions & Spacing")
        dims_layout = QGridLayout()
        dims_layout.setSpacing(8)

        dims_layout.addWidget(QLabel("Original XY Size:"), 0, 0)
        self.orig_xy_size = QDoubleSpinBox()
        self.orig_xy_size.setRange(0.000001, 10000.0)
        self.orig_xy_size.setValue(1.0)
        self.orig_xy_size.setDecimals(6)
        self.orig_xy_size.setToolTip("Original pixel size in XY plane.")
        dims_layout.addWidget(self.orig_xy_size, 0, 1)

        dims_layout.addWidget(QLabel("Original Z Spacing:"), 1, 0)
        self.orig_z_size = QDoubleSpinBox()
        self.orig_z_size.setRange(0.000001, 10000.0)
        self.orig_z_size.setValue(1.0)
        self.orig_z_size.setDecimals(6)
        self.orig_z_size.setToolTip("Original distance between slices.")
        dims_layout.addWidget(self.orig_z_size, 1, 1)

        dims_layout.addWidget(QLabel("New XY Size:"), 0, 2)
        self.new_xy_size = QDoubleSpinBox()
        self.new_xy_size.setRange(0.000001, 10000.0)
        self.new_xy_size.setValue(1.0)
        self.new_xy_size.setDecimals(6)
        self.new_xy_size.setToolTip("Desired output pixel size in XY plane.")
        dims_layout.addWidget(self.new_xy_size, 0, 3)

        dims_layout.addWidget(QLabel("New Z Spacing:"), 1, 2)
        self.new_z_size = QDoubleSpinBox()
        self.new_z_size.setRange(0.000001, 10000.0)
        self.new_z_size.setValue(1.0)
        self.new_z_size.setDecimals(6)
        self.new_z_size.setToolTip("Desired output distance between slices.")
        dims_layout.addWidget(self.new_z_size, 1, 3)

        dims_layout.addWidget(QLabel("Unit:"), 2, 0)
        self.size_unit = QComboBox()
        self.size_unit.addItems(["nm", "µm", "mm"])
        self.size_unit.setCurrentText("µm")
        self.size_unit.setToolTip("Unit for all size and spacing values.")
        dims_layout.addWidget(self.size_unit, 2, 1)

        dims_group.setLayout(dims_layout)
        layout.addWidget(dims_group)


        # --- Output Selection ---
        output_group = QGroupBox("Output Location")
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory: Not selected")
        self.output_label.setWordWrap(True)
        self.select_output_btn = QPushButton("Select...")
        self.select_output_btn.setToolTip("Select directory to save the interpolated stack")
        self.select_output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(self.select_output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Action Button & Progress ---
        self.interpolate_btn = QPushButton("Interpolate Stack")
        self.interpolate_btn.clicked.connect(self.interpolate_stack)
        layout.addWidget(self.interpolate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle") # Initial text
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def select_input(self):
        """Selects input directory or TIFF stack file."""
        try:
            if self.dir_radio.isChecked():
                path = QFileDialog.getExistingDirectory(
                    self, "Select Directory with Images", ""
                )
                dialog_title = "Select Directory"
            else: # Stack file
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select TIFF Stack File", "", "TIFF Files (*.tif *.tiff)"
                )
                dialog_title = "Select TIFF Stack"

            if path:
                self.input_path = path
                self.input_label.setText(f"Input: ...{os.sep}{os.path.basename(path)}")
                self.input_label.setToolTip(path)
                QApplication.processEvents()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting input: {str(e)}")

    def select_output(self):
        """Selects the output directory."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", ""
            )
            if directory:
                self.output_directory = directory
                self.output_label.setText(f"Output: ...{os.sep}{os.path.basename(directory)}")
                self.output_label.setToolTip(directory)
                QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting output directory: {str(e)}")

    def load_images(self, progress_dialog):
        """Loads images from directory or stack file, handling potential errors."""
        print("Starting image loading...")
        try:
            if self.stack_radio.isChecked():
                progress_dialog.setLabelText("Loading TIFF stack...")
                QApplication.processEvents()
                stack = io.imread(self.input_path)
                if stack.ndim < 3: raise ValueError("Selected TIFF is not a multi-page stack.")
                print(f"Loaded TIFF stack shape: {stack.shape}, dtype: {stack.dtype}")
                return stack
            else: # Directory of images
                progress_dialog.setLabelText("Loading images from directory...")
                QApplication.processEvents()
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                # Sort files naturally if possible
                try:
                    import natsort
                    files = natsort.natsorted([f for f in os.listdir(self.input_path)
                                             if f.lower().endswith(valid_extensions)])
                    print("Using natural sorting for files.")
                except ImportError:
                    files = sorted([f for f in os.listdir(self.input_path)
                                  if f.lower().endswith(valid_extensions)])
                    print("Warning: 'natsort' package not found. Using simple alphabetical sorting.")

                if not files: raise ValueError("No valid image files found in directory.")

                progress_dialog.setMaximum(len(files))
                progress_dialog.setValue(0)

                # Load first image to get shape and dtype
                first_path = os.path.join(self.input_path, files[0])
                first_img = io.imread(first_path)
                ref_shape = first_img.shape
                ref_dtype = first_img.dtype
                print(f"Reference image '{files[0]}': shape={ref_shape}, dtype={ref_dtype}")

                # Initialize stack
                stack = np.zeros((len(files), *ref_shape), dtype=ref_dtype)
                stack[0] = first_img

                # Load remaining images
                for i, fname in enumerate(files[1:], 1):
                    if progress_dialog.wasCanceled(): raise InterruptedError("Loading cancelled")
                    progress_dialog.setValue(i)
                    progress_dialog.setLabelText(f"Loading image {i+1}/{len(files)}: {fname}")
                    QApplication.processEvents()

                    img_path = os.path.join(self.input_path, fname)
                    img = io.imread(img_path)
                    if img.shape != ref_shape:
                        raise ValueError(f"Image '{fname}' shape {img.shape} differs from first image {ref_shape}.")
                    if img.dtype != ref_dtype:
                         print(f"Warning: Image '{fname}' dtype {img.dtype} differs from first image {ref_dtype}. Attempting conversion.")
                         try: img = img.astype(ref_dtype)
                         except (ValueError, TypeError) as conv_err: raise ValueError(f"Cannot convert {fname} to {ref_dtype}: {conv_err}")
                    stack[i] = img

                print(f"Loaded stack from directory: shape={stack.shape}, dtype={stack.dtype}")
                return stack

        except Exception as e:
            print(f"Error during image loading: {e}")
            traceback.print_exc()
            raise ValueError(f"Error loading images: {str(e)}") # Re-raise with generic message

    def interpolate_stack(self):
        """Performs the stack interpolation based on UI settings."""
        if not self.input_path or not self.output_directory:
            QMessageBox.warning(self, "Missing Paths", "Please select both input path and output directory.")
            return
        if self.orig_xy_size.value() <= 0 or self.orig_z_size.value() <= 0 or \
           self.new_xy_size.value() <= 0 or self.new_z_size.value() <= 0:
           QMessageBox.warning(self, "Invalid Spacing", "Pixel sizes and Z spacing must be positive.")
           return

        progress = None
        try:
            # --- Setup ---
            progress = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal) # Use enum
            progress.setWindowTitle("Interpolation Progress")
            progress.setMinimumDuration(0) # Show immediately
            progress.setMinimumWidth(400)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            # --- Load Images ---
            input_stack = self.load_images(progress) # Pass progress dialog
            if input_stack.ndim != 3: # Expecting ZYX
                 raise ValueError(f"Input stack must be 3-dimensional (Z, Y, X). Got shape {input_stack.shape}")

            original_dtype = input_stack.dtype
            print(f"Original stack: dtype={original_dtype}, shape={input_stack.shape}, range=[{np.min(input_stack)}, {np.max(input_stack)}]")

            # --- Prepare Coordinates and Interpolator ---
            progress.setLabelText("Preparing interpolation grid...")
            progress.setValue(20)
            QApplication.processEvents()

            # Original coordinates based on original spacing
            z_orig_coords = np.arange(input_stack.shape[0]) * self.orig_z_size.value()
            y_orig_coords = np.arange(input_stack.shape[1]) * self.orig_xy_size.value()
            x_orig_coords = np.arange(input_stack.shape[2]) * self.orig_xy_size.value()

            # New coordinates based on new spacing, covering the same *spatial* range
            # Add a small epsilon to ensure the last original point is included in the range
            epsilon = 1e-6 * max(self.new_z_size.value(), self.new_xy_size.value())
            z_new_coords = np.arange(z_orig_coords[0], z_orig_coords[-1] + epsilon, self.new_z_size.value())
            y_new_coords = np.arange(y_orig_coords[0], y_orig_coords[-1] + epsilon, self.new_xy_size.value())
            x_new_coords = np.arange(x_orig_coords[0], x_orig_coords[-1] + epsilon, self.new_xy_size.value())

            new_shape = (len(z_new_coords), len(y_new_coords), len(x_new_coords))
            if np.prod(new_shape) == 0: raise ValueError("New dimensions result in zero size.")
            print(f"Original grid covers Z: [{z_orig_coords[0]}, {z_orig_coords[-1]}], Y: [{y_orig_coords[0]}, {y_orig_coords[-1]}], X: [{x_orig_coords[0]}, {x_orig_coords[-1]}]")
            print(f"New grid covers Z: [{z_new_coords[0]}, {z_new_coords[-1]}], Y: [{y_new_coords[0]}, {y_new_coords[-1]}], X: [{x_new_coords[0]}, {x_new_coords[-1]}]")
            print(f"New dimensions will be: {new_shape}")

            # Normalize input data to float64 for potentially better interpolation precision
            input_stack_float = input_stack.astype(np.float64)
            input_min = np.min(input_stack_float)
            input_max = np.max(input_stack_float)
            # Avoid scaling if range is zero
            input_range = input_max - input_min
            if input_range > 1e-9: # Use tolerance for float comparison
                 input_stack_float = (input_stack_float - input_min) / input_range
            else:
                 print("Warning: Input stack has zero range. Output will be constant.")
                 input_stack_float.fill(0.0) # Set to 0 if flat


            # Choose interpolation method
            method = self.method_combo.currentText()
            print(f"Using interpolation method: {method}")

            # --- Perform Interpolation ---
            progress.setLabelText(f"Interpolating using '{method}' method...")
            progress.setValue(40)
            QApplication.processEvents()

            # Create the interpolator function
            # Use lower-case 'linear' and 'nearest' for scipy
            scipy_method = method if method in ['linear', 'nearest'] else 'linear' # Default to linear for griddata issues
            if method not in ['linear', 'nearest']:
                print(f"Note: Scipy RegularGridInterpolator only directly supports 'linear' and 'nearest'. Using 'linear' for grid interpolation.")
                # Hybrid approach from previous attempt is very complex and slow, sticking to RegularGridInterpolator limits

            interpolator = RegularGridInterpolator(
                (z_orig_coords, y_orig_coords, x_orig_coords),
                input_stack_float, # Use normalized float data
                method=scipy_method,
                bounds_error=False, # Don't raise error for points outside original grid
                fill_value=0 # Fill outside points with background (0 after normalization)
            )

            # Create the grid of new points to interpolate at
            # meshgrid creates coordinate matrices matching the *output* shape
            zz_new, yy_new, xx_new = np.meshgrid(
                z_new_coords, y_new_coords, x_new_coords, indexing='ij' # 'ij' indexing -> output matches (Z, Y, X) order
            )
            # Stack the coordinate matrices into a (N, 3) array where N = Z*Y*X (new)
            points_to_interpolate = np.stack([zz_new.ravel(), yy_new.ravel(), xx_new.ravel()], axis=-1)

            # Perform interpolation (this can be memory intensive for large outputs)
            print(f"Interpolating {points_to_interpolate.shape[0]} points...")
            interpolated_data_flat = interpolator(points_to_interpolate)

            # Reshape the flat result back to the target 3D shape
            interpolated_data_float = interpolated_data_flat.reshape(new_shape)
            print(f"Interpolation finished. Result shape: {interpolated_data_float.shape}")

            # --- Convert back to original dtype range ---
            progress.setLabelText("Converting to original data type...")
            progress.setValue(80)
            QApplication.processEvents()

            final_array = interpolated_data_float # Start with float result

            if np.issubdtype(original_dtype, np.integer):
                print(f"Scaling result back to original dtype ({original_dtype}) range...")
                type_info = np.iinfo(original_dtype)
                # Scale normalized [0, 1] result back to original min/max range
                # If input range was zero, output remains zero (scaled)
                if input_range > 1e-9:
                     final_array = final_array * input_range + input_min
                else: # Handle flat input case
                     final_array.fill(input_min) # Fill with the original constant value

                # Clip to ensure values are within the target type's bounds before casting
                final_array = np.clip(final_array, type_info.min, type_info.max)
                final_array = final_array.astype(original_dtype)
                print(f"Final integer data range: [{np.min(final_array)}, {np.max(final_array)}]")
            elif np.issubdtype(original_dtype, np.floating):
                 # Scale back if needed (or just cast if original was float)
                 if input_range > 1e-9:
                     final_array = final_array * input_range + input_min
                 else:
                     final_array.fill(input_min)
                 final_array = final_array.astype(original_dtype)
                 print(f"Final float data range: [{np.min(final_array)}, {np.max(final_array)}]")
            else:
                 print(f"Warning: Unsupported original dtype {original_dtype}. Saving result as float64.")
                 final_array = final_array.astype(np.float64) # Keep as float64

            # --- Save Output TIFF with Metadata ---
            progress.setLabelText("Saving interpolated stack...")
            progress.setValue(95)
            QApplication.processEvents()

            # Determine output filename
            if self.stack_radio.isChecked():
                output_name_base = os.path.splitext(os.path.basename(self.input_path))[0]
            else:
                output_name_base = f"interpolated_{os.path.basename(self.input_path)}"
            output_filename = f"{output_name_base}_interpolated_{method}.tif" # Include method in name
            output_path = os.path.join(self.output_directory, output_filename)

            # Get pixel sizes in micrometers for metadata
            unit = self.size_unit.currentText()
            xy_size_um = self.new_xy_size.value()
            z_size_um = self.new_z_size.value()
            if unit == "nm": xy_size_um /= 1000.0; z_size_um /= 1000.0
            elif unit == "mm": xy_size_um *= 1000.0; z_size_um *= 1000.0

            # Prepare ImageJ metadata
            imagej_metadata = {'axes': 'ZYX', 'unit': 'um'}
            if z_size_um > 0: imagej_metadata['spacing'] = z_size_um
            resolution_xy = (1.0 / xy_size_um, 1.0 / xy_size_um) if xy_size_um > 0 else (1.0, 1.0)
            resolution_unit = 'micron'

            print(f"Saving interpolated stack to: {output_path}")
            print(f"  Metadata: {imagej_metadata}")
            print(f"  Resolution: {resolution_xy} pixels/{resolution_unit}")

            # Save the stack
            tifffile.imwrite(
                output_path,
                final_array,
                imagej=True,
                metadata=imagej_metadata,
                resolution=resolution_xy,
                resolutionunit=resolution_unit
            )

            if progress: progress.setValue(100)
            QApplication.processEvents()

            QMessageBox.information(
                self,
                "Success",
                f"Interpolation completed successfully!\n"
                f"Output saved to:\n{output_path}\n\n"
                f"Original Shape: {input_stack.shape}\n"
                f"New Shape: {final_array.shape}\n"
                f"Output Bit Depth: {final_array.dtype}\n"
                f"Output XY Pixel Size: {self.new_xy_size.value():.4f} {unit}\n"
                f"Output Z Spacing: {self.new_z_size.value():.4f} {unit}"
            )

        except InterruptedError:
             QMessageBox.warning(self, "Cancelled", "Operation cancelled by user.")
             if progress: progress.setLabelText("Cancelled.")
        except Exception as e:
            error_msg = f"An error occurred during interpolation:\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
            traceback.print_exc()
            if progress: progress.setLabelText("Error occurred.")
        finally:
             if progress: progress.close()
             QApplication.processEvents()

    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            if self.rect().isValid(): # Check if self geometry is valid
                 self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()

# Helper function to create the dialog
def show_stack_interpolator(parent):
    """Creates and shows the Stack Interpolator dialog."""
    dialog = StackInterpolator(parent)
    dialog.show_centered(parent)
    return dialog # Return instance if needed
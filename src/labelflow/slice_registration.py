# slice_registration.py (Migrated to PyQt6)
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                             QLabel, QComboBox, QMessageBox, QProgressDialog, QRadioButton,
                             QButtonGroup, QSpinBox, QApplication, QGroupBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt
from pystackreg import StackReg # Assuming pystackreg is compatible
from skimage import io # Assuming scikit-image is compatible
import tifffile
from PIL import Image # Used indirectly via skimage.io potentially
import numpy as np
import os
import traceback # For error logging

class SliceRegistrationTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice Registration Tool")
        self.setGeometry(100, 100, 600, 500) # Adjusted height
        # Use PyQt6 enums
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Initialize variables first
        self.input_path = ""
        self.output_directory = ""

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # --- Input Selection ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()

        self.dir_radio = QRadioButton("Directory of Image Files (Sorted Numerically)")
        self.stack_radio = QRadioButton("Multi-Page TIFF Stack")
        self.input_type_group = QButtonGroup(self) # Group radio buttons
        self.input_type_group.addButton(self.dir_radio)
        self.input_type_group.addButton(self.stack_radio)
        self.dir_radio.setChecked(True) # Default to directory
        input_layout.addWidget(self.dir_radio)
        input_layout.addWidget(self.stack_radio)

        file_select_layout = QHBoxLayout()
        self.input_label = QLabel("Input: Not selected")
        self.input_label.setWordWrap(True)
        self.select_input_btn = QPushButton("Select...")
        self.select_input_btn.setToolTip("Select input directory or TIFF stack file")
        self.select_input_btn.clicked.connect(self.select_input)
        file_select_layout.addWidget(self.input_label, 1) # Allow label to stretch
        file_select_layout.addWidget(self.select_input_btn)
        input_layout.addLayout(file_select_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Output Selection ---
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory: Not selected")
        self.output_label.setWordWrap(True)
        self.select_output_btn = QPushButton("Select...")
        self.select_output_btn.setToolTip("Select directory to save the registered stack")
        self.select_output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label, 1) # Allow label to stretch
        output_layout.addWidget(self.select_output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Registration Settings ---
        settings_group = QGroupBox("Registration Settings")
        settings_layout = QGridLayout() # Use grid for alignment
        settings_layout.setSpacing(8)

        settings_layout.addWidget(QLabel("Transformation:"), 0, 0)
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Translation", # Simpler names
            "Rigid Body",
            "Scaled Rotation",
            "Affine",
            "Bilinear"
        ])
        self.transform_combo.setToolTip("Select the type of transformation allowed for registration.")
        settings_layout.addWidget(self.transform_combo, 0, 1, 1, 3) # Span columns

        settings_layout.addWidget(QLabel("Reference Frame:"), 1, 0)
        self.ref_combo = QComboBox()
        self.ref_combo.addItems([
            "Previous", # Simpler names
            "First",
            "Mean",
            "Mean of First N",
            "Mean + Moving Average"
        ])
        self.ref_combo.setToolTip("Select the reference frame to which other slices will be registered.")
        self.ref_combo.currentTextChanged.connect(self.on_ref_changed) # Connect signal
        settings_layout.addWidget(self.ref_combo, 1, 1, 1, 3) # Span columns

        # N frames settings (conditionally enabled)
        settings_layout.addWidget(QLabel("N Frames (for Mean):"), 2, 0)
        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(1, 1000) # Increased range
        self.n_frames_spin.setValue(10)
        self.n_frames_spin.setEnabled(False) # Disabled initially
        self.n_frames_spin.setToolTip("Number of initial frames to use for 'Mean of First N' references.")
        settings_layout.addWidget(self.n_frames_spin, 2, 1)

        # Moving average settings (conditionally enabled)
        settings_layout.addWidget(QLabel("Moving Avg Window:"), 3, 0)
        self.moving_avg_spin = QSpinBox()
        self.moving_avg_spin.setRange(1, 1000) # Increased range
        self.moving_avg_spin.setValue(10)
        self.moving_avg_spin.setEnabled(False) # Disabled initially
        self.moving_avg_spin.setToolTip("Window size for 'Mean + Moving Average' reference.")
        settings_layout.addWidget(self.moving_avg_spin, 3, 1)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Pixel/Voxel Size (for Metadata) ---
        spacing_group = QGroupBox("Output Metadata (Pixel/Voxel Size)")
        spacing_layout = QGridLayout()
        spacing_layout.setSpacing(8)

        spacing_layout.addWidget(QLabel("XY Pixel Size:"), 0, 0)
        self.xy_size_value = QDoubleSpinBox()
        self.xy_size_value.setRange(0.000001, 10000.0)
        self.xy_size_value.setValue(1.0)
        self.xy_size_value.setDecimals(6)
        self.xy_size_value.setToolTip("Size of a pixel in X and Y (used for output TIFF metadata).")
        spacing_layout.addWidget(self.xy_size_value, 0, 1)

        spacing_layout.addWidget(QLabel("Z Spacing:"), 1, 0)
        self.z_size_value = QDoubleSpinBox()
        self.z_size_value.setRange(0.000001, 10000.0)
        self.z_size_value.setValue(1.0)
        self.z_size_value.setDecimals(6)
        self.z_size_value.setToolTip("Distance between slices (used for output TIFF metadata).")
        spacing_layout.addWidget(self.z_size_value, 1, 1)

        spacing_layout.addWidget(QLabel("Unit:"), 0, 2)
        self.size_unit = QComboBox()
        self.size_unit.addItems(["nm", "µm", "mm"])
        self.size_unit.setCurrentText("µm")
        self.size_unit.setToolTip("Unit for XY and Z sizes.")
        spacing_layout.addWidget(self.size_unit, 0, 3)

        spacing_group.setLayout(spacing_layout)
        layout.addWidget(spacing_group)

        # --- Action Button ---
        self.register_btn = QPushButton("Register Stack")
        self.register_btn.clicked.connect(self.register_slices)
        layout.addWidget(self.register_btn)

        self.setLayout(layout)
        self.on_ref_changed(self.ref_combo.currentText()) # Initial setup of spinbox enable state

    def on_ref_changed(self, text):
        """Enables/disables spin boxes based on selected reference type."""
        uses_n_frames = text in ["Mean of First N", "Mean + Moving Average"]
        uses_moving_avg = text == "Mean + Moving Average"
        self.n_frames_spin.setEnabled(uses_n_frames)
        self.moving_avg_spin.setEnabled(uses_moving_avg)
        QApplication.processEvents()

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
        """Loads images from directory or stack file."""
        try:
            if self.stack_radio.isChecked():
                progress_dialog.setLabelText("Loading TIFF stack...")
                QApplication.processEvents()
                # Load stack using scikit-image (handles various TIFF types)
                stack = io.imread(self.input_path)
                print(f"Loaded TIFF stack shape: {stack.shape}, dtype: {stack.dtype}")
                return stack
            else: # Directory of images
                progress_dialog.setLabelText("Loading images from directory...")
                QApplication.processEvents()
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                # Sort files naturally (e.g., img1, img2, img10) if possible, else alphabetically
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
                         # Try to convert if possible (e.g., uint8 to uint16 might lose info)
                         print(f"Warning: Image '{fname}' dtype {img.dtype} differs from first image {ref_dtype}. Attempting conversion.")
                         try:
                              img = img.astype(ref_dtype)
                         except (ValueError, TypeError) as conv_err:
                              raise ValueError(f"Could not convert image '{fname}' to match dtype {ref_dtype}: {conv_err}")

                    stack[i] = img

                print(f"Loaded stack from directory: shape={stack.shape}, dtype={stack.dtype}")
                return stack

        except Exception as e:
            print(f"Error during image loading: {e}")
            traceback.print_exc()
            raise # Re-raise to be caught by register_slices

    def register_slices(self):
        """Performs the slice registration."""
        if not self.input_path or not self.output_directory:
            QMessageBox.warning(self, "Input/Output Missing", "Please select both input and output paths.")
            return

        progress = None
        try:
            progress = QProgressDialog(self)
            progress.setWindowTitle("Registration Progress")
            progress.setLabelText("Initializing...")
            progress.setMinimum(0)
            progress.setMaximum(100) # Percentage based
            progress.setValue(0)
            progress.setWindowModality(Qt.WindowModality.WindowModal) # Use enum
            progress.setMinimumWidth(400)
            progress.setMinimumDuration(0) # Show immediately
            progress.show()
            QApplication.processEvents()

            # Load images
            img_stack = self.load_images(progress) # Pass progress dialog
            if img_stack is None or img_stack.ndim < 3: # Need at least 3D (Z, Y, X)
                 raise ValueError("Loaded data is not a valid stack (needs at least 3 dimensions).")

            original_dtype = img_stack.dtype
            print(f"Input stack properties: dtype={original_dtype}, shape={img_stack.shape}, range=[{np.min(img_stack)}, {np.max(img_stack)}]")

            # --- Perform Registration ---
            progress.setValue(30)
            progress.setLabelText("Performing registration...")
            QApplication.processEvents()

            # Map UI names to pystackreg constants/reference strings
            transform_map = {
                "Translation": StackReg.TRANSLATION,
                "Rigid Body": StackReg.RIGID_BODY,
                "Scaled Rotation": StackReg.SCALED_ROTATION,
                "Affine": StackReg.AFFINE,
                "Bilinear": StackReg.BILINEAR
            }
            reference_map = {
                "Previous": 'previous',
                "First": 'first',
                "Mean": 'mean'
                # 'Mean of First N' and 'Mean + Moving Average' handled below
            }

            selected_transform = transform_map[self.transform_combo.currentText()]
            selected_ref_text = self.ref_combo.currentText()

            sr = StackReg(selected_transform)

            # Prepare arguments for registration based on reference type
            reg_kwargs = {}
            if selected_ref_text in reference_map:
                reg_kwargs['reference'] = reference_map[selected_ref_text]
            elif selected_ref_text == "Mean of First N":
                reg_kwargs['reference'] = 'first' # Base reference is first
                reg_kwargs['n_frames'] = self.n_frames_spin.value()
            elif selected_ref_text == "Mean + Moving Average":
                reg_kwargs['reference'] = 'first' # Base reference is first
                reg_kwargs['n_frames'] = self.n_frames_spin.value()
                reg_kwargs['moving_average'] = self.moving_avg_spin.value()
            else:
                raise ValueError(f"Unknown reference type: {selected_ref_text}")

            print(f"Registering with transform='{self.transform_combo.currentText()}', kwargs={reg_kwargs}")

            # Convert to float32 for registration calculations if not already float
            if not np.issubdtype(img_stack.dtype, np.floating):
                 print("Converting stack to float32 for registration...")
                 img_stack_float = img_stack.astype(np.float32)
            else:
                 img_stack_float = img_stack

            # pystackreg modifies the input array *in place* when transforming
            # To keep the original, make a copy before passing to register_transform_stack
            # However, register_transform_stack *returns* the transformed stack, so maybe copy isn't needed?
            # Let's assume it returns a new array based on common practice. Check pystackreg docs if issues.
            # registered_stack_float = sr.register_transform_stack(img_stack_float.copy(), **reg_kwargs)
            registered_stack_float = sr.register_transform_stack(img_stack_float, **reg_kwargs)


            progress.setValue(80)
            progress.setLabelText("Saving registered stack...")
            QApplication.processEvents()

            # --- Convert back to original dtype range ---
            final_stack = registered_stack_float
            if np.issubdtype(original_dtype, np.integer):
                 print(f"Converting registered stack back to original dtype ({original_dtype}) range...")
                 type_info = np.iinfo(original_dtype)
                 data_min = final_stack.min()
                 data_max = final_stack.max()
                 if data_max > data_min:
                      final_stack = (final_stack - data_min) / (data_max - data_min)
                      final_stack = final_stack * (type_info.max - type_info.min) + type_info.min
                 else: # Flat result
                      final_stack = np.full(final_stack.shape, type_info.min, dtype=final_stack.dtype)
                 final_stack = np.clip(final_stack, type_info.min, type_info.max).astype(original_dtype)
                 print(f"Final integer data range: [{np.min(final_stack)}, {np.max(final_stack)}]")
            elif np.issubdtype(original_dtype, np.floating):
                 final_stack = final_stack.astype(original_dtype)
                 print(f"Final float data range: [{np.min(final_stack)}, {np.max(final_stack)}]")
            else:
                 print(f"Warning: Could not convert back to original dtype {original_dtype}. Saving as float32.")
                 final_stack = final_stack.astype(np.float32)

            print(f"Final output stack: dtype={final_stack.dtype}, shape={final_stack.shape}")

            # --- Save Output TIFF with Metadata ---
            if self.stack_radio.isChecked():
                output_name_base = os.path.splitext(os.path.basename(self.input_path))[0]
            else:
                output_name_base = f"registered_stack_{os.path.basename(self.input_path)}" # Include dir name
            output_filename = f"{output_name_base}_registered.tif"
            output_path = os.path.join(self.output_directory, output_filename)

            # Get pixel sizes in micrometers
            unit = self.size_unit.currentText()
            xy_size_um = self.xy_size_value.value()
            z_size_um = self.z_size_value.value()
            if unit == "nm": xy_size_um /= 1000.0; z_size_um /= 1000.0
            elif unit == "mm": xy_size_um *= 1000.0; z_size_um *= 1000.0

            # Prepare ImageJ metadata
            imagej_metadata = {'axes': 'ZYX', 'unit': 'um'} # Assume ZYX for 3D stack
            if z_size_um > 0: imagej_metadata['spacing'] = z_size_um
            # Use resolution argument for XY
            resolution_xy = (1.0 / xy_size_um, 1.0 / xy_size_um) if xy_size_um > 0 else (1.0, 1.0)
            resolution_unit = 'micron'

            print(f"Saving registered stack to: {output_path}")
            print(f"  Metadata: {imagej_metadata}")
            print(f"  Resolution: {resolution_xy} pixels/{resolution_unit}")

            tifffile.imwrite(
                output_path,
                final_stack,
                imagej=True,
                metadata=imagej_metadata,
                resolution=resolution_xy,
                resolutionunit=resolution_unit
            )

            progress.setValue(100)
            QApplication.processEvents()

            QMessageBox.information(self, "Success",
                                  f"Registration completed successfully!\n"
                                  f"Output saved to:\n{output_path}\n"
                                  f"Output XY Pixel size: {self.xy_size_value.value():.4f} {unit}\n"
                                  f"Output Z Spacing: {self.z_size_value.value():.4f} {unit}")

        except InterruptedError:
             QMessageBox.warning(self, "Cancelled", "Operation cancelled by user.")
        except Exception as e:
            error_msg = f"An error occurred during registration:\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
            traceback.print_exc()
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
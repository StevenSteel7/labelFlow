# dicom_converter.py (Migrated to PyQt6)
import os
import json
import numpy as np
from datetime import datetime
import traceback  # Import traceback for detailed error printing

# PyQt6 Imports
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                             QLabel, QProgressDialog, QRadioButton, QButtonGroup,
                             QMessageBox, QApplication, QGroupBox, QDoubleSpinBox, QComboBox,
                             QGridLayout) # Added QGridLayout
from PyQt6.QtCore import Qt

# External library imports
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tifffile

class DicomConverter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM to TIFF Converter")
        self.setGeometry(100, 100, 600, 450) # Adjusted initial height
        # Use Qt.WindowType and Qt.WindowModality enums
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Initialize variables first
        self.input_file = ""
        self.output_directory = ""

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # --- Input/Output Selection ---
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_label = QLabel("No DICOM file selected")
        self.input_label.setWordWrap(True)
        self.select_input_btn = QPushButton("Select DICOM File")
        self.select_input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.select_input_btn)
        input_layout.addWidget(self.input_label, 1) # Allow label to stretch
        file_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_label = QLabel("No output directory selected")
        self.output_label.setWordWrap(True)
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.select_output_btn)
        output_layout.addWidget(self.output_label, 1) # Allow label to stretch
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- Output Format ---
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout()
        self.stack_radio = QRadioButton("Single TIFF Stack")
        self.individual_radio = QRadioButton("Individual TIFF Files")
        self.stack_radio.setChecked(True)
        self.format_button_group = QButtonGroup(self)
        self.format_button_group.addButton(self.stack_radio)
        self.format_button_group.addButton(self.individual_radio)
        format_layout.addWidget(self.stack_radio)
        format_layout.addWidget(self.individual_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # --- Pixel/Voxel Size Input ---
        spacing_group = QGroupBox("Pixel/Voxel Size (for Metadata)")
        spacing_layout = QGridLayout() # Use GridLayout for better alignment

        spacing_layout.addWidget(QLabel("XY Pixel Size:"), 0, 0)
        self.xy_size_value = QDoubleSpinBox()
        self.xy_size_value.setRange(0.000001, 10000.0) # Allow smaller values
        self.xy_size_value.setValue(1.0)
        self.xy_size_value.setDecimals(6) # Allow more precision
        self.xy_size_value.setToolTip("Size of a pixel in the X and Y dimensions.")
        spacing_layout.addWidget(self.xy_size_value, 0, 1)

        spacing_layout.addWidget(QLabel("Z Spacing:"), 1, 0)
        self.z_size_value = QDoubleSpinBox()
        self.z_size_value.setRange(0.000001, 10000.0) # Allow smaller values
        self.z_size_value.setValue(1.0)
        self.z_size_value.setDecimals(6) # Allow more precision
        self.z_size_value.setToolTip("Distance between slices in the Z dimension.")
        spacing_layout.addWidget(self.z_size_value, 1, 1)

        spacing_layout.addWidget(QLabel("Unit:"), 2, 0)
        self.size_unit = QComboBox()
        self.size_unit.addItems(["nm", "µm", "mm"])
        self.size_unit.setCurrentText("µm")
        self.size_unit.setToolTip("Unit for the XY and Z sizes.")
        spacing_layout.addWidget(self.size_unit, 2, 1)

        spacing_group.setLayout(spacing_layout)
        layout.addWidget(spacing_group)

        # --- Metadata Info ---
        metadata_group = QGroupBox("Metadata Information")
        metadata_layout = QVBoxLayout()
        metadata_label = QLabel("Relevant DICOM metadata will be saved as a JSON file alongside the TIFF output.")
        metadata_label.setStyleSheet("color: gray; font-style: italic;")
        metadata_label.setWordWrap(True)
        metadata_layout.addWidget(metadata_label)
        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)

        # --- Convert Button ---
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.convert_dicom)
        layout.addWidget(self.convert_btn)

        self.setLayout(layout)

    def select_input(self):
        """Opens a file dialog to select the input DICOM file."""
        try:
            file_filter = "DICOM files (*.dcm *.DCM);;All files (*.*)"
            # QFileDialog.getOpenFileName returns tuple (fileName, selectedFilter)
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select DICOM File", "", file_filter
            )
            if file_name:
                self.input_file = file_name
                self.input_label.setText(self.truncate_path(file_name))
                self.input_label.setToolTip(file_name)
                QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting input file: {str(e)}")

    def select_output(self):
        """Opens a directory dialog to select the output folder."""
        try:
            # QFileDialog.getExistingDirectory returns a string path
            directory = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", ""
            )
            if directory:
                self.output_directory = directory
                self.output_label.setText(self.truncate_path(directory))
                self.output_label.setToolTip(directory)
                QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting output directory: {str(e)}")

    def truncate_path(self, path, max_length=50):
        """Shortens a path string for display in the UI."""
        if len(path) <= max_length:
            return path
        parts = path.split(os.sep)
        if len(parts) <= 1: # If no separators, just truncate
            return "..." + path[-(max_length-3):]
        filename = parts[-1]
        if len(filename) >= max_length - 5: # Filename itself is too long
            return "..." + filename[-(max_length - 5):]
        # Build path backwards
        truncated_path = filename
        current_len = len(filename)
        for i in range(len(parts) - 2, -1, -1):
            part = parts[i]
            if current_len + len(part) + 5 <= max_length: # +5 for separator and "..."
                truncated_path = os.path.join(part, truncated_path)
                current_len += len(part) + 1
            else:
                truncated_path = os.path.join("...", truncated_path)
                break
        return truncated_path

    def extract_metadata(self, ds):
        """Extracts relevant metadata from a pydicom dataset."""
        def get_str(tag, default="Unknown"):
            val = getattr(ds, tag, default)
            if isinstance(val, pydicom.multival.MultiValue):
                 # Join multi-value sequences, limit length if needed
                 str_vals = [str(v) for v in val]
                 joined = " / ".join(str_vals)
                 return joined[:200] + "..." if len(joined) > 200 else joined # Limit length
            elif isinstance(val, pydicom.valuerep.PersonName):
                 return str(val)
            elif isinstance(val, bytes):
                 try: return val.decode('utf-8', errors='replace').strip()
                 except Exception: return f"<{len(val)} bytes>" # Placeholder for unreadable bytes
            return str(val).strip() # General case

        def get_num_list(tag, default_val=[1.0, 1.0]):
            val = getattr(ds, tag, None)
            if isinstance(val, pydicom.multival.MultiValue):
                try: return [float(v) for v in val]
                except (ValueError, TypeError): return default_val
            elif isinstance(val, (int, float, str)):
                try: return [float(val)] # Return as list
                except (ValueError, TypeError): return default_val
            return default_val

        def get_num(tag, default_val=1.0):
            val = getattr(ds, tag, None)
            # Handle multi-value cases by taking the first element
            if isinstance(val, pydicom.multival.MultiValue):
                if len(val) > 0: val = val[0]
                else: val = None # Treat empty MultiValue as None
            # Try to convert to float
            if val is not None:
                try: return float(val)
                except (ValueError, TypeError): return default_val
            return default_val

        metadata = {
            "PatientID": get_str("PatientID"),
            "PatientName": get_str("PatientName"),
            "StudyDate": get_str("StudyDate"),
            "SeriesDescription": get_str("SeriesDescription"),
            "Modality": get_str("Modality"),
            "Manufacturer": get_str("Manufacturer"),
            "InstitutionName": get_str("InstitutionName"),
            "PixelSpacing": get_num_list("PixelSpacing", [1.0, 1.0]),
            "SliceThickness": get_num("SliceThickness", 1.0),
            "ImageOrientationPatient": get_num_list("ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "ImagePositionPatient": get_num_list("ImagePositionPatient", [0.0, 0.0, 0.0]),
            "WindowCenter": get_num("WindowCenter", None), # Let apply_window_level handle None
            "WindowWidth": get_num("WindowWidth", None),   # Let apply_window_level handle None
            "RescaleIntercept": get_num("RescaleIntercept", 0.0),
            "RescaleSlope": get_num("RescaleSlope", 1.0),
            "BitsAllocated": int(get_num("BitsAllocated", 16)),
            "PixelRepresentation": int(get_num("PixelRepresentation", 0)), # 0=unsigned, 1=signed
            "PhotometricInterpretation": get_str("PhotometricInterpretation", ""),
            "Rows": int(get_num("Rows", 0)),
            "Columns": int(get_num("Columns", 0)),
            "SamplesPerPixel": int(get_num("SamplesPerPixel", 1)),
            "NumberOfFrames": int(get_num("NumberOfFrames", 1)),
            "ConversionDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return metadata

    def apply_window_level(self, image, ds):
        """Applies VOI LUT (window/level) if present in DICOM."""
        try:
            # Ensure image is float for potentially better precision with VOI LUT
            if image.dtype != np.float32 and image.dtype != np.float64:
                 image = image.astype(np.float64) # Use float64 for potentially large values
            # pydicom's function handles presence check and potential VOI LUT Sequence
            # prefer_lut=True uses VOI LUT Sequence entry first if available
            return apply_voi_lut(image, ds, prefer_lut=True)
        except Exception as e:
            print(f"Could not apply VOI LUT: {e}. Returning original image.")
            return image # Return original (but possibly float converted) image

    def convert_dicom(self):
        """Reads the DICOM, processes, and saves as TIFF."""
        if not self.input_file or not self.output_directory:
            QMessageBox.warning(self, "Input/Output Missing", "Please select both input file and output directory.")
            return

        os.makedirs(self.output_directory, exist_ok=True)
        progress = None # Initialize progress to None
        try:
            # Create progress dialog
            progress = QProgressDialog("Processing DICOM file...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumWidth(400)
            progress.setMinimumDuration(0) # Show immediately
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            # --- Read DICOM ---
            progress.setLabelText("Reading DICOM file...")
            progress.setValue(5)
            QApplication.processEvents()
            try:
                ds = pydicom.dcmread(self.input_file, force=True)
            except Exception as read_err:
                 raise ValueError(f"Failed to read DICOM file: {read_err}")

            series_metadata = self.extract_metadata(ds)

            # --- Process Pixel Data ---
            progress.setLabelText("Processing pixel data...")
            progress.setValue(20)
            QApplication.processEvents()
            if not hasattr(ds, 'PixelData'):
                raise ValueError("DICOM file does not contain PixelData.")

            pixel_array = ds.pixel_array
            original_dtype = pixel_array.dtype
            print(f"Original data: dtype={original_dtype}, shape={pixel_array.shape}, range=[{np.min(pixel_array)}, {np.max(pixel_array)}]")

            # Apply Rescale Slope/Intercept
            slope = series_metadata.get("RescaleSlope", 1.0)
            intercept = series_metadata.get("RescaleIntercept", 0.0)
            if slope != 1.0 or intercept != 0.0:
                print(f"Applying rescale slope ({slope}) and intercept ({intercept})")
                pixel_array = pixel_array.astype(np.float64) * slope + intercept
            else:
                 # Ensure float for potential windowing even if no rescale
                 pixel_array = pixel_array.astype(np.float64)

            # Apply Window/Level (VOI LUT)
            print("Applying window/level adjustments (if available)...")
            pixel_array_windowed = self.apply_window_level(pixel_array, ds)
            print(f"Windowed data range: [{np.min(pixel_array_windowed)}, {np.max(pixel_array_windowed)}]")

            # --- Save Metadata ---
            progress.setLabelText("Saving metadata...")
            progress.setValue(60)
            QApplication.processEvents()
            metadata_filename = os.path.splitext(os.path.basename(self.input_file))[0] + "_metadata.json"
            metadata_file = os.path.join(self.output_directory, metadata_filename)
            try:
                # Convert numpy types specifically, handle others with default=str
                def default_serializer(obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                    if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
                    if isinstance(obj, (np.bool_)): return bool(obj)
                    if isinstance(obj, (np.void)): return None # Or other representation
                    return str(obj) # Fallback to string for unknown types

                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(series_metadata, f, indent=2, default=default_serializer)
                print(f"Metadata saved to {metadata_file}")
            except Exception as json_err:
                print(f"Warning: Could not save metadata as JSON: {json_err}")

            # --- Prepare final array for saving ---
            # Normalize windowed data to the original data type's range
            final_array = pixel_array_windowed
            if np.issubdtype(original_dtype, np.integer):
                print(f"Normalizing windowed data to original dtype ({original_dtype}) range...")
                type_info = np.iinfo(original_dtype)
                data_min = final_array.min()
                data_max = final_array.max()
                if data_max > data_min: # Avoid division by zero
                    final_array = (final_array - data_min) / (data_max - data_min)
                    final_array = final_array * (type_info.max - type_info.min) + type_info.min
                else: # Handle flat image case
                    final_array = np.full(final_array.shape, type_info.min, dtype=np.float64)
                # Clip to ensure values are within the target type's bounds before casting
                final_array = np.clip(final_array, type_info.min, type_info.max)
                final_array = final_array.astype(original_dtype)
                print(f"Final integer data range: [{np.min(final_array)}, {np.max(final_array)}]")
            elif np.issubdtype(original_dtype, np.floating):
                 final_array = final_array.astype(original_dtype)
                 print(f"Final float data range: [{np.min(final_array)}, {np.max(final_array)}]")
            else:
                 print(f"Warning: Unsupported original dtype {original_dtype}. Saving as windowed float64.")
                 # Keep as float64 if unsure

            # --- Prepare ImageJ Metadata ---
            unit = self.size_unit.currentText()
            xy_size_um = self.xy_size_value.value()
            z_size_um = self.z_size_value.value()
            if unit == "nm": xy_size_um /= 1000.0; z_size_um /= 1000.0
            elif unit == "mm": xy_size_um *= 1000.0; z_size_um *= 1000.0

            if final_array.ndim >= 3: axes = 'ZYX'[:final_array.ndim][::-1] # ZYX, TZYX etc.
            elif final_array.ndim == 2: axes = 'YX'
            else: axes = 'X'; print(f"Warning: Unexpected array dimension {final_array.ndim}")

            imagej_metadata = {'axes': axes, 'unit': 'um'}
            if 'Z' in axes: imagej_metadata['spacing'] = z_size_um
            # Use resolution argument for XY spacing
            resolution_xy = (1.0 / xy_size_um, 1.0 / xy_size_um) if xy_size_um > 0 else (1.0, 1.0)
            resolution_unit = 'micron'

            print(f"ImageJ metadata: {imagej_metadata}, resolution: {resolution_xy} pixels/{resolution_unit}")

            # --- Save TIFF ---
            progress.setLabelText("Saving TIFF file(s)...")
            progress.setValue(80)
            QApplication.processEvents()
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            saved_files = []

            if self.stack_radio.isChecked():
                output_file = os.path.join(self.output_directory, f"{base_name}.tif")
                tifffile.imwrite(output_file, final_array, imagej=True, metadata=imagej_metadata,
                                 resolution=resolution_xy, resolutionunit=resolution_unit)
                print(f"Saved stack: {output_file} (shape: {final_array.shape})")
                saved_files.append(os.path.basename(output_file))
            else: # Individual slices
                if final_array.ndim >= 3:
                    total_slices = final_array.shape[0]
                    slice_metadata = imagej_metadata.copy()
                    slice_metadata['axes'] = 'YX' # Slices are 2D
                    if 'spacing' in slice_metadata: del slice_metadata['spacing']

                    progress.setLabelText(f"Saving slice 1/{total_slices}...")
                    progress.setMaximum(total_slices) # Update max for per-slice progress
                    progress.setValue(0)
                    QApplication.processEvents()

                    for i in range(total_slices):
                        if progress.wasCanceled(): raise InterruptedError("Saving cancelled")
                        slice_filename = f"{base_name}_slice_{i+1:04d}.tif" # Use 4 digits padding
                        output_file = os.path.join(self.output_directory, slice_filename)
                        tifffile.imwrite(output_file, final_array[i], imagej=True, metadata=slice_metadata,
                                         resolution=resolution_xy, resolutionunit=resolution_unit)
                        saved_files.append(slice_filename)
                        progress.setValue(i + 1)
                        progress.setLabelText(f"Saving slice {i+1}/{total_slices}...")
                        QApplication.processEvents()
                    print(f"Saved {total_slices} individual slices.")
                else: # Input was 2D
                    output_file = os.path.join(self.output_directory, f"{base_name}.tif")
                    tifffile.imwrite(output_file, final_array, imagej=True, metadata=imagej_metadata,
                                     resolution=resolution_xy, resolutionunit=resolution_unit)
                    print(f"Saved single slice: {output_file} (shape: {final_array.shape})")
                    saved_files.append(os.path.basename(output_file))

            if progress: progress.setValue(100) # Ensure progress completes
            QApplication.processEvents()

            # --- Success Message ---
            msg = "Conversion complete!\n\n"
            msg += f"DICOM file: {os.path.basename(self.input_file)}\n"
            msg += f"Output directory: {self.truncate_path(self.output_directory)}\n\n"
            if self.stack_radio.isChecked():
                msg += f"Saved stack as: {saved_files[0]}\n"
            else:
                if final_array.ndim >= 3: msg += f"Saved {len(saved_files)} individual slices\n"
                else: msg += f"Saved slice as: {saved_files[0]}\n"
            if metadata_file and os.path.exists(metadata_file): msg += f"Metadata saved as: {metadata_filename}\n"
            msg += f"Output XY Pixel size: {self.xy_size_value.value():.4f} {unit}\n"
            if final_array.ndim >= 3: msg += f"Output Z Spacing: {self.z_size_value.value():.4f} {unit}"
            QMessageBox.information(self, "Success", msg)

        except InterruptedError as e:
             QMessageBox.warning(self, "Cancelled", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during conversion:\n{str(e)}")
            print(f"Error occurred: {str(e)}")
            traceback.print_exc() # Print detailed traceback
        finally:
             if progress: progress.close() # Ensure progress dialog is closed
             QApplication.processEvents()

    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            # Ensure self has valid geometry before calculating center
            if self.rect().isValid():
                 self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()  # Ensure window displays properly

# Function to show the dialog
def show_dicom_converter(parent):
    """Creates and shows the DICOM Converter dialog."""
    dialog = DicomConverter(parent)
    dialog.show_centered(parent)
    return dialog # Return instance if needed
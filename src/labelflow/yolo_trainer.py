# yolo_trainer.py (Migrated to PyQt6 - Corrected Syntax & Formatting)
import os
import traceback # For detailed error logging
import random # For temporary file naming
import shutil # For copying files

# PyQt6 Imports
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, \
                            QPushButton, QLineEdit, QLabel, QDialogButtonBox, QTextEdit
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# External Libraries
from ultralytics import YOLO
import yaml
import numpy as np
from pathlib import Path
from collections import deque

# Local Imports (Ensure these are PyQt6 compatible or don't use Qt)
# Assuming export_yolo_v5plus is correctly defined elsewhere
from .export_formats import export_yolo_v5plus


class TrainingInfoDialog(QDialog):
    """Dialog to show training progress information."""
    stop_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Training Progress")
        self.setModal(False) # Allow interaction with main window
        # Use Qt.WindowType enum
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        layout = QVBoxLayout(self)

        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)
        self.info_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Optional: prevent wrap
        layout.addWidget(self.info_text)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.setToolTip("Request training to stop after the current epoch.")
        self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.setToolTip("Close this progress window (training continues).")
        self.close_button.clicked.connect(self.hide) # Just hide, don't close/destroy
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.setMinimumSize(500, 400) # Adjusted size

    def update_info(self, text):
        """Appends text to the info display."""
        self.info_text.append(text)
        # Auto-scroll to the bottom
        self.info_text.verticalScrollBar().setValue(self.info_text.verticalScrollBar().maximum())

    def stop_training(self):
        """Emits signal to stop training and updates button state."""
        self.stop_signal.emit()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping...")

    def reset_dialog(self):
        """Resets the dialog state for a new training run."""
        self.info_text.clear()
        self.stop_button.setEnabled(True)
        self.stop_button.setText("Stop Training")

    def closeEvent(self, event):
        """Override close event to just hide the dialog."""
        event.ignore() # Prevent actual closing
        self.hide()

class LoadPredictionModelDialog(QDialog):
    """Dialog to select YOLO model (.pt) and corresponding YAML file."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Prediction Model & YAML")
        # Use Qt.WindowType enum
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setModal(True) # Make modal
        self.model_path = ""
        self.yaml_path = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select .pt model file")
        model_button = QPushButton("Browse...")
        model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("Model File:"))
        model_layout.addWidget(self.model_edit, 1) # Allow edit to stretch
        model_layout.addWidget(model_button)
        layout.addLayout(model_layout)

        # YAML file selection
        yaml_layout = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("Select .yaml dataset file")
        yaml_button = QPushButton("Browse...")
        yaml_button.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(QLabel("YAML File:"))
        yaml_layout.addWidget(self.yaml_edit, 1) # Allow edit to stretch
        yaml_layout.addWidget(yaml_button)
        layout.addLayout(yaml_layout)

        # OK and Cancel buttons
        # Use QDialogButtonBox standard buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel) # Use enum
        self.button_box.accepted.connect(self.accept) # Connect to QDialog's accept
        self.button_box.rejected.connect(self.reject) # Connect to QDialog's reject
        layout.addWidget(self.button_box)

    def browse_model(self):
        """Opens file dialog to select a YOLO model file."""
        # QFileDialog.getOpenFileName returns tuple (fileName, selectedFilter)
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if file_name:
            self.model_path = file_name
            self.model_edit.setText(file_name)
            self.model_edit.setToolTip(file_name) # Show full path on hover

    def browse_yaml(self):
        """Opens file dialog to select a YAML configuration file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_name:
            self.yaml_path = file_name
            self.yaml_edit.setText(file_name)
            self.yaml_edit.setToolTip(file_name) # Show full path on hover

# --- YOLOTrainer Class ---
class YOLOTrainer(QObject):
    """Handles YOLO model loading, training, prediction, and dataset preparation."""
    progress_signal = pyqtSignal(str) # Emits training progress strings

    def __init__(self, project_dir, main_window):
        super().__init__()
        self.project_dir = project_dir
        self.main_window = main_window # Keep reference to main window
        self.model = None # The loaded/trained YOLO model instance
        self.dataset_path = os.path.join(project_dir, "yolo_dataset") # Default export path
        self.model_save_dir = os.path.join(project_dir, "yolo_models") # Default save path for runs
        self.yaml_path = None # Path to the loaded dataset YAML
        self.yaml_data = None # Loaded content of the YAML
        self.prediction_yaml = None # YAML loaded specifically for prediction
        self.class_names = None # Class names from the prediction YAML
        self.epoch_info = deque(maxlen=15) # Store more recent epoch info
        self.progress_callback = None # External callback function for progress (optional)
        self.total_epochs = None
        self.conf_threshold = 0.25 # Default prediction confidence
        self.stop_training = False # Flag to signal training stop

    def load_model(self, model_path=None):
        """Loads a pre-trained YOLO model from a .pt file."""
        if model_path is None:
            model_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Model (.pt)", "", "YOLO Model (*.pt)")
        if model_path:
            try:
                print(f"Loading YOLO model from: {model_path}")
                self.model = YOLO(model_path)
                print("Model loaded successfully.")
                return True
            except Exception as e:
                error_msg = f"Could not load the model.\nError: {str(e)}"
                QMessageBox.critical(self.main_window, "Error Loading Model", error_msg)
                print(error_msg)
                traceback.print_exc()
        return False

    def prepare_dataset(self):
        """Exports current annotations to YOLOv5+ format AND copies train data to val."""
        print("Preparing YOLO dataset...")
        if not self.main_window.all_annotations:
             raise ValueError("No annotations available to export for YOLO dataset.")
        if not self.main_window.class_mapping:
             raise ValueError("No classes defined to export for YOLO dataset.")

        # Ensure dataset base path exists
        os.makedirs(self.dataset_path, exist_ok=True)

        try:
            # Step 1: Export data using the standard YOLOv5+ structure
            # This function should create dataset_path/images/train, dataset_path/labels/train,
            # dataset_path/images/val, dataset_path/labels/val and dataset_path/data.yaml
            output_dir, yaml_path = export_yolo_v5plus(
                self.main_window.all_annotations,
                self.main_window.class_mapping,
                self.main_window.image_paths,
                self.main_window.slices,
                self.main_window.image_slices,
                self.dataset_path # Export to the designated dataset path
            )
            print(f"YOLO dataset exported to: {output_dir}")
            print(f"Dataset YAML created at: {yaml_path}")

            # --- Step 2: Copy Train data to Val directories (Workaround for validation error) ---
            # This ensures Ultralytics finds files even if validating on the training set
            images_train_dir = os.path.join(output_dir, 'images', 'train')
            labels_train_dir = os.path.join(output_dir, 'labels', 'train')
            images_val_dir = os.path.join(output_dir, 'images', 'val')
            labels_val_dir = os.path.join(output_dir, 'labels', 'val')

            # Ensure val directories exist (export should have created them, but be safe)
            os.makedirs(images_val_dir, exist_ok=True)
            os.makedirs(labels_val_dir, exist_ok=True)

            print("Copying training data to validation directories for compatibility...")
            copy_errors = 0
            # Copy images
            if os.path.isdir(images_train_dir):
                 train_img_files = os.listdir(images_train_dir)
                 if train_img_files:
                      for f in train_img_files:
                           try: shutil.copy2(os.path.join(images_train_dir, f), images_val_dir)
                           except Exception as cp_err: print(f"Warn: Failed copy {f} to val images: {cp_err}"); copy_errors+=1
                 else: print("Warning: Training image directory is empty.")
            else: print(f"Warning: Training image directory not found: {images_train_dir}")

            # Copy labels
            if os.path.isdir(labels_train_dir):
                 train_lbl_files = os.listdir(labels_train_dir)
                 if train_lbl_files:
                      for f in train_lbl_files:
                           try: shutil.copy2(os.path.join(labels_train_dir, f), labels_val_dir)
                           except Exception as cp_err: print(f"Warn: Failed copy {f} to val labels: {cp_err}"); copy_errors+=1
                 else: print("Warning: Training label directory is empty.")
            else: print(f"Warning: Training label directory not found: {labels_train_dir}")

            if copy_errors == 0: print("Finished copying training data to validation directories.")
            else: print(f"Finished copying training data to validation directories with {copy_errors} errors.")
            # --- End Copy Step ---

            # Update internal yaml_path after successful export/preparation
            self.yaml_path = str(yaml_path)
            # Load the generated YAML data immediately
            self.load_yaml(self.yaml_path)
            return self.yaml_path

        except Exception as e:
             error_msg = f"Failed to prepare YOLO dataset: {e}"
             print(error_msg)
             traceback.print_exc()
             raise ValueError(error_msg) from e


    def load_yaml(self, yaml_path=None):
        """Loads and processes paths in a dataset YAML file."""
        if yaml_path is None:
            yaml_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)")
        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f: # Specify encoding
                    yaml_data = yaml.safe_load(f)
                print(f"Loaded YAML contents from {yaml_path}: {yaml_data}")

                # --- Path Handling ---
                yaml_dir = os.path.dirname(yaml_path)
                root_path = yaml_data.get('path', yaml_dir) # Default to YAML dir if 'path' missing
                if not os.path.isabs(root_path):
                    root_path = os.path.abspath(os.path.join(yaml_dir, root_path))
                yaml_data['path'] = root_path # Store absolute path

                # Ensure train/val/test paths are relative to the 'path' key
                for key in ['train', 'val', 'test']:
                    if key in yaml_data and yaml_data[key]: # Check if key exists and has a value
                         p_rel = yaml_data[key]
                         # Check if path is already relative (heuristic)
                         is_already_relative = not os.path.isabs(p_rel) and not Path(p_rel).drive

                         if not is_already_relative:
                              # If it was absolute, make it relative to root_path
                              abs_p = os.path.abspath(p_rel) # Ensure it's absolute first
                              try:
                                   yaml_data[key] = os.path.relpath(abs_p, start=root_path)
                              except ValueError as e:
                                   print(f"Warning: Could not make path relative for '{key}' (possibly different drives): {e}. Keeping absolute path: {abs_p}")
                                   yaml_data[key] = abs_p # Keep absolute if relpath fails
                         # else: path is already relative, assume it's relative to root_path

                         # Verify the *final* absolute path exists after resolving
                         final_abs_path = os.path.abspath(os.path.join(root_path, yaml_data[key]))
                         if not os.path.exists(final_abs_path):
                              print(f"Warning: Path for '{key}' ('{yaml_data[key]}' relative to '{root_path}') does not exist: {final_abs_path}")


                print(f"Processed YAML contents for training: {yaml_data}")
                self.yaml_data = yaml_data
                self.yaml_path = yaml_path
                return True

            except yaml.YAMLError as e:
                QMessageBox.critical(self.main_window, "YAML Error", f"Invalid YAML file: {yaml_path}.\nError: {str(e)}")
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error Loading YAML", f"Could not load or process YAML file: {yaml_path}.\nError: {str(e)}")
                traceback.print_exc()
        return False

    def on_train_epoch_end(self, trainer):
        """Callback function executed by Ultralytics at the end of each training epoch."""
        try:
            # Access attributes defensively
            epoch = getattr(trainer, 'epoch', -1) + 1 # epoch is 0-based
            total_epochs = getattr(trainer, 'epochs', self.total_epochs) # total epochs
            loss = getattr(trainer, 'loss', None) # Combined loss tensor

            # Extract specific metrics if available
            metrics = getattr(trainer, 'metrics', {})
            box_loss = metrics.get('train/box_loss', float('nan'))
            cls_loss = metrics.get('train/cls_loss', float('nan'))
            dfl_loss = metrics.get('train/dfl_loss', float('nan'))
            seg_loss = metrics.get('train/seg_loss', float('nan')) # Add seg loss if applicable
            val_map = metrics.get('metrics/mAP50-95(B)', float('nan')) # Box mAP
            val_map50 = metrics.get('metrics/mAP50(B)', float('nan'))   # Box mAP@0.5
            val_map_seg = metrics.get('metrics/mAP50-95(M)', float('nan'))# Mask mAP
            val_map50_seg = metrics.get('metrics/mAP50(M)', float('nan')) # Mask mAP@0.5

            loss_val = loss.item() if loss is not None else float('nan')

            # Format progress string
            progress_text = f"Epoch {epoch}/{total_epochs} | Loss: {loss_val:.4f} "
            loss_parts = []
            if not np.isnan(box_loss): loss_parts.append(f"Box:{box_loss:.3f}")
            if not np.isnan(cls_loss): loss_parts.append(f"Cls:{cls_loss:.3f}")
            if not np.isnan(seg_loss): loss_parts.append(f"Seg:{seg_loss:.3f}")
            if not np.isnan(dfl_loss): loss_parts.append(f"DFL:{dfl_loss:.3f}")
            if loss_parts: progress_text += f"({', '.join(loss_parts)}) | "
            progress_text += f"Val mAP50(B):{val_map50:.3f} | Val mAP50(M):{val_map50_seg:.3f}"

            # Emit signal for UI update
            self.progress_signal.emit(progress_text)

            # Check stop flag
            if self.stop_training:
                print("Stop request received by callback.")
                # Attempt to stop the trainer gracefully
                if hasattr(trainer, 'stop'): trainer.stop = True
                if hasattr(trainer, 'model') and hasattr(trainer.model, 'stop'): trainer.model.stop = True
                self.stop_training = False # Reset flag
                self.progress_signal.emit("Training stopping...")
                # Returning False might not work reliably; flag is primary mechanism

        except Exception as e:
            print(f"Error in training callback: {e}")
            traceback.print_exc()

        return True # Tell Ultralytics to continue training


    def train_model(self, epochs=100, imgsz=640):
        """Starts the YOLO model training process using the loaded/prepared YAML."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        if self.yaml_path is None or not Path(self.yaml_path).exists():
            print("YAML path not found or invalid. Attempting to prepare dataset...")
            try:
                self.prepare_dataset()
                if self.yaml_path is None or not Path(self.yaml_path).exists():
                     raise FileNotFoundError("Dataset YAML still not found after preparation.")
            except Exception as prep_err:
                 raise FileNotFoundError(f"Dataset YAML not found and preparation failed: {prep_err}") from prep_err

        self.stop_training = False
        self.total_epochs = epochs
        self.epoch_info.clear()

        # Add the callback using the documented method
        if "on_train_epoch_end" not in self.model.callbacks:
             self.model.callbacks["on_train_epoch_end"] = []
        if self.on_train_epoch_end not in self.model.callbacks["on_train_epoch_end"]:
             self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

        # Ensure model save directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)

        try:
            yaml_path_str = str(self.yaml_path) # Use the prepared/loaded YAML directly
            print(f"Starting training with YAML: {yaml_path_str}, Epochs: {epochs}, ImgSz: {imgsz}")
            print(f"Model save directory (project): {self.model_save_dir}")

            # Ultralytics training call
            results = self.model.train(
                data=yaml_path_str, # Pass the path to the main data.yaml
                epochs=epochs,
                imgsz=imgsz,
                project=self.model_save_dir, # Main dir for runs
                name='train',               # Specific run subdir name
                exist_ok=True               # Allow overwriting previous 'train' run
                # device=0                  # Optional: specify GPU device (e.g., 0, 'cpu')
            )
            print("Training finished.")
            return results
        except FileNotFoundError as fnf_err:
             error_msg = f"Dataset structure error or file not found:\n{fnf_err}\n\nPlease ensure the paths in your YAML are correct and the directories exist."
             print(error_msg)
             QMessageBox.critical(self.main_window, "Dataset Error", error_msg)
             raise # Re-raise after showing message
        except Exception as e:
             error_msg = f"An unexpected error occurred during training:\n{e}"
             print(error_msg)
             traceback.print_exc()
             QMessageBox.critical(self.main_window, "Training Error", error_msg)
             raise # Re-raise other exceptions
        finally:
            # Clean up callback after training finishes or errors out
            if self.model and "on_train_epoch_end" in self.model.callbacks:
                 try:
                      self.model.callbacks["on_train_epoch_end"].remove(self.on_train_epoch_end)
                 except ValueError:
                      pass # Ignore if already removed


    def stop_training_signal(self):
        """Slot to be connected to the Stop button's signal."""
        self.stop_training = True
        self.progress_signal.emit(">>> Stop request received. Training will stop after current epoch.")

    def set_progress_callback(self, callback):
        """Sets an external function to receive progress updates (if needed beyond signal)."""
        self.progress_callback = callback # Not currently used if signal is sufficient

    def save_model(self):
        """Saves a *copy* of the best or last trained model weights."""
        # Check if a model exists and if it has potentially been trained
        if self.model is None:
             QMessageBox.warning(self.main_window, "No Model", "No model loaded or trained yet.")
             return False
        if not hasattr(self.model, 'trainer') or not hasattr(self.model.trainer, 'save_dir'):
             QMessageBox.warning(self.main_window, "Model Not Trained", "Model has not been trained in this session, or results directory is unknown. Cannot save.")
             return False

        # Find the path to the weights from the last training run
        run_dir = Path(self.model.trainer.save_dir)
        best_weights = run_dir / 'weights' / 'best.pt'
        last_weights = run_dir / 'weights' / 'last.pt'
        source_path = None

        if best_weights.exists():
            source_path = best_weights
            print(f"Found best weights at: {source_path}")
        elif last_weights.exists():
            source_path = last_weights
            print(f"Found last weights at: {source_path}")
        else:
             QMessageBox.warning(self.main_window, "No Weights Found", f"Could not find 'best.pt' or 'last.pt' in the run directory:\n{run_dir}")
             return False

        # Ask user where to save a *copy* of the weights
        suggested_name = f"{run_dir.name}_{source_path.name}" # e.g., train_best.pt
        save_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Trained YOLO Model As",
            os.path.join(self.project_dir, suggested_name), # Suggest saving in project root
            "YOLO Model (*.pt)"
        )
        if save_path:
            try:
                # Ensure suffix is .pt
                if not save_path.lower().endswith(".pt"):
                    save_path += ".pt"
                shutil.copy2(source_path, save_path) # Copy the weight file
                QMessageBox.information(self.main_window, "Model Saved", f"Model weights saved to:\n{save_path}")
                print(f"Model copied from {source_path} to {save_path}")
                return True
            except Exception as e:
                error_msg = f"Could not copy model weights.\nError: {str(e)}"
                QMessageBox.critical(self.main_window, "Error Saving Model", error_msg)
                print(error_msg)
        return False

    def load_prediction_model(self, model_path, yaml_path):
        """Loads a model and its corresponding YAML for prediction tasks."""
        try:
            # Load the model
            print(f"Loading prediction model from: {model_path}")
            self.model = YOLO(model_path)

            # Load the prediction YAML
            print(f"Loading prediction YAML from: {yaml_path}")
            with open(yaml_path, 'r', encoding='utf-8') as f: # Specify encoding
                self.prediction_yaml = yaml.safe_load(f)

            # Validate YAML structure
            if 'names' not in self.prediction_yaml or not isinstance(self.prediction_yaml['names'], list):
                raise ValueError("Prediction YAML file must contain a 'names' list for class names.")

            self.class_names = self.prediction_yaml['names']
            print(f"Loaded class names for prediction: {self.class_names}")

            # --- Class Name / Count Verification ---
            model_class_names = self.model.names # Access model's internal names/classes (dict {idx: name})
            num_yaml_classes = len(self.class_names)
            num_model_classes = len(model_class_names)

            mismatch_message = ""
            if num_yaml_classes != num_model_classes:
                mismatch_message = (f"Warning: Number of classes mismatch!\n"
                                    f"YAML has {num_yaml_classes} names: {self.class_names}\n"
                                    f"Model expects {num_model_classes} classes: {list(model_class_names.values())}\n"
                                    "Predictions might use incorrect class labels.")
                print(mismatch_message)
            # Optional: Check if names actually match, not just count
            # else:
            #     yaml_name_dict = {i: name for i, name in enumerate(self.class_names)}
            #     if yaml_name_dict != model_class_names:
            #         print("Warning: Class names in YAML do not exactly match model names, but count is the same.")
            #         # Optionally update model names?
            #         # self.model.names = yaml_name_dict

            print("Prediction model and YAML loaded successfully.")
            return True, mismatch_message # Return success and any warning message

        except Exception as e:
            error_message = f"Error loading model or YAML for prediction.\nError: {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.model = None # Reset model on failure
            self.prediction_yaml = None
            self.class_names = None
            return False, error_message # Return failure and error message


    def predict(self, input_data):
        """Runs prediction (segmentation) on input image (path or numpy array)."""
        if self.model is None:
            raise ValueError("No prediction model loaded. Please use 'Load Model' first.")
        if self.class_names is None:
             raise ValueError("Prediction class names not loaded. Ensure YAML was loaded correctly with the prediction model.")

        try:
             # Run prediction using the loaded model
             print(f"Running prediction with confidence threshold: {self.conf_threshold}")
             results = self.model.predict(
                 source=input_data,
                 task='segment', # Ensure segmentation task
                 conf=self.conf_threshold,
                 save=False, # Don't save images/labels automatically
                 show=False, # Don't display results automatically
                 verbose=False # Reduce console output during prediction
             )

             if not results:
                  print("Prediction returned no results.")
                  # Return format consistent with success case, but empty
                  return [], (0,0), (0,0) # Empty list, zero shapes

             # Ultralytics predict returns a list of Results objects (usually one for single image)
             result = results[0]

             # Extract necessary info from the Results object
             input_size = result.orig_shape # Shape of the original image fed to model
             # Attempt to get processed size, fallback to input_size
             processed_size = result.speed.get('preprocess', input_size) if hasattr(result, 'speed') and isinstance(result.speed, dict) else input_size

             print(f"Prediction successful. Original shape: {input_size}, Processed shape (approx): {processed_size}")
             # Return the list of results, original shape, original shape (matching previous expectation)
             return results, input_size, input_size

        except Exception as e:
             print(f"Error during YOLO prediction: {e}")
             traceback.print_exc()
             raise # Re-raise the exception to be handled by the caller


    def set_conf_threshold(self, conf):
        """Sets the confidence threshold for predictions."""
        if 0.0 <= conf <= 1.0:
            self.conf_threshold = conf
            print(f"Prediction confidence threshold set to: {self.conf_threshold}")
        else:
            print(f"Warning: Invalid confidence threshold {conf}. Must be between 0.0 and 1.0.")
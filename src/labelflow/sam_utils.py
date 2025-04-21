# sam_utils.py (Migrated to PyQt6)

import numpy as np
# PyQt6 Imports
from PyQt6.QtGui import QImage, QColor
from ultralytics import SAM
import cv2 # OpenCV needed for mask_to_polygon and potentially format conversions
import traceback # For detailed error logging

class SAMUtils:
    def __init__(self):
        self.sam_models = {
            "SAM 2 tiny": "sam2_t.pt",
            "SAM 2 small": "sam2_s.pt",
            "SAM 2 base": "sam2_b.pt",
            "SAM 2 large": "sam2_l.pt"
        }
        self.current_sam_model = None
        self.sam_model = None

    def change_sam_model(self, model_name):
        """Loads the specified SAM model."""
        if model_name != "Pick a SAM Model":
            self.current_sam_model = model_name
            model_file = self.sam_models.get(self.current_sam_model)
            if model_file:
                try:
                    print(f"Loading SAM model: {model_file}...")
                    # Ensure the model file exists or handle potential download
                    # Note: Ultralytics SAM might handle download automatically if not found locally
                    self.sam_model = SAM(model_file)
                    print(f"Successfully loaded SAM model: {model_name}")
                except Exception as e:
                    print(f"Error initializing SAM model {model_name} ({model_file}): {e}")
                    traceback.print_exc()
                    self.current_sam_model = None
                    self.sam_model = None
                    # Optionally raise or show a message to the user
                    # raise RuntimeError(f"Failed to load SAM model: {e}") from e
            else:
                 print(f"Error: Model file not defined for '{model_name}'.")
                 self.current_sam_model = None
                 self.sam_model = None
        else:
            self.current_sam_model = None
            self.sam_model = None
            print("SAM model unset")

    def normalize_16bit_to_8bit(self, array):
        """Normalizes a 16-bit numpy array to 8-bit (uint8)."""
        # Handle potential division by zero if max == min
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val == min_val:
            # Return a zero array or an array filled with the min value scaled?
            # For visualization, a mid-gray might be better if min/max are non-zero
            # return np.full_like(array, 128, dtype=np.uint8)
            return np.zeros_like(array, dtype=np.uint8) # Simple zero array for now

        # Perform normalization safely using float64 for intermediate calculations
        normalized = (array.astype(np.float64) - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)

    def qimage_to_numpy(self, qimage: QImage) -> np.ndarray:
        """
        Converts a QImage to a NumPy array (BGR format, uint8).
        Handles various QImage formats and potential memory padding.
        """
        if qimage.isNull():
            raise ValueError("Cannot convert a null QImage.")

        img_format = qimage.format()
        ptr = qimage.constBits() # Get pointer to raw data

        # Check if pointer is valid
        if ptr is None:
             # If constBits() returns None, try bits() if modification is okay (less ideal)
             # ptr = qimage.bits()
             # if ptr is None:
             raise ValueError("Could not get QImage bits (constBits returned None).")

        # Get buffer size safely
        num_bytes = qimage.sizeInBytes()
        if num_bytes <= 0:
            raise ValueError(f"Invalid QImage size: {num_bytes} bytes for format {img_format}.")

        # Access buffer data using memoryview for safety
        # Note: sip.voidptr doesn't directly support buffer protocol in all versions/configs.
        # Accessing its internal buffer might require specific casting or methods
        # depending on exact PyQt/sip version. Using `asarray` if available is safer.
        try:
            # Check if asarray method exists (safer access)
            buf = ptr.asarray(num_bytes)
        except AttributeError:
             # Fallback: If asarray not available, try to interpret pointer directly (less safe)
             # This might require ctypes or other methods depending on exact PyQt/sip version
             # For now, raise an error indicating incompatibility if asarray is missing.
             print("Warning: ptr.asarray(num_bytes) method not available. Direct buffer access might be needed.")
             # Example using ctypes (highly platform/version dependent):
             # import ctypes
             # buffer_ptr = ctypes.cast(int(ptr), ctypes.POINTER(ctypes.c_ubyte))
             # buf = np.ctypeslib.as_array(buffer_ptr, shape=(num_bytes,))
             # This fallback is complex and fragile, better to ensure compatible PyQt/sip version.
             raise NotImplementedError("Cannot access QImage buffer: ptr.asarray() missing and fallback not implemented.")

        arr = np.frombuffer(buf, dtype=np.uint8)

        height = qimage.height()
        width = qimage.width()
        bytes_per_line = qimage.bytesPerLine()

        # Reshape array considering potential padding (bytes_per_line)
        expected_bytes_8bit = width
        expected_bytes_16bit = width * 2
        expected_bytes_rgb888 = width * 3
        expected_bytes_rgba32 = width * 4

        # --- Handle different formats ---
        if img_format == QImage.Format.Format_RGB32 or \
           img_format == QImage.Format.Format_ARGB32 or \
           img_format == QImage.Format.Format_ARGB32_Premultiplied:
            if bytes_per_line >= expected_bytes_rgba32:
                arr = arr.reshape((height, bytes_per_line))
                arr = arr[:, :expected_bytes_rgba32] # Slice to remove padding
                arr = arr.reshape((height, width, 4))
                return arr[:, :, :3][:, :, ::-1] # Slice alpha, Convert BGRA to BGR
            else: raise ValueError(f"Unexpected bytes per line ({bytes_per_line}) for RGBA32 format (expected >= {expected_bytes_rgba32})")

        elif img_format == QImage.Format.Format_RGB888:
            if bytes_per_line >= expected_bytes_rgb888:
                arr = arr.reshape((height, bytes_per_line))
                arr = arr[:, :expected_bytes_rgb888] # Slice padding
                arr = arr.reshape((height, width, 3))
                return arr[:, :, ::-1] # Convert RGB to BGR
            else: raise ValueError(f"Unexpected bytes per line ({bytes_per_line}) for RGB888 format (expected >= {expected_bytes_rgb888})")

        elif img_format == QImage.Format.Format_Grayscale8:
            if bytes_per_line >= expected_bytes_8bit:
                arr = arr.reshape((height, bytes_per_line))
                arr = arr[:, :expected_bytes_8bit] # Slice padding
                arr = arr.reshape((height, width))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else: raise ValueError(f"Unexpected bytes per line ({bytes_per_line}) for Grayscale8 format (expected >= {expected_bytes_8bit})")

        elif img_format == QImage.Format.Format_Grayscale16:
            # Need to reinterpret buffer as uint16
            if num_bytes % 2 != 0: raise ValueError("Grayscale16 should have an even number of bytes.")
            buf_u16 = np.frombuffer(buf, dtype=np.uint16)
            if bytes_per_line >= expected_bytes_16bit:
                 buf_u16 = buf_u16.reshape((height, bytes_per_line // 2))
                 buf_u16 = buf_u16[:, :width] # Slice padding
                 arr16 = buf_u16.reshape((height, width))
                 arr8 = self.normalize_16bit_to_8bit(arr16)
                 return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
            else: raise ValueError(f"Unexpected bytes per line ({bytes_per_line}) for Grayscale16 format (expected >= {expected_bytes_16bit})")

        elif img_format == QImage.Format.Format_Indexed8:
            # Slower path for indexed images
            if bytes_per_line >= expected_bytes_8bit:
                 arr = arr.reshape((height, bytes_per_line))
                 arr = arr[:, :expected_bytes_8bit] # Slice padding
                 index_arr = arr.reshape((height, width)) # This is the index array

                 color_table = qimage.colorTable()
                 if not color_table: raise ValueError("Indexed8 image missing color table.")
                 # Create an empty BGR image
                 bgr_image = np.zeros((height, width, 3), dtype=np.uint8)
                 # Map indices to colors (more efficient numpy way)
                 colors_rgb = np.array([QColor(c).getRgb()[:3] for c in color_table], dtype=np.uint8)
                 bgr_colors = colors_rgb[:, ::-1] # Convert RGB table to BGR table
                 # Use fancy indexing
                 bgr_image = bgr_colors[index_arr]
                 return bgr_image
            else: raise ValueError(f"Unexpected bytes per line ({bytes_per_line}) for Indexed8 format (expected >= {expected_bytes_8bit})")

        else:
            # Fallback: convert to a known format (prefer RGB888 for color)
            target_format = QImage.Format.Format_RGB888
            print(f"Warning: Unsupported QImage format {img_format}. Attempting conversion to {target_format}.")
            converted_image = qimage.convertToFormat(target_format)
            if converted_image.isNull():
                raise ValueError(f"Could not convert QImage from format {img_format} to {target_format}.")
            # Recursive call with the converted image
            return self.qimage_to_numpy(converted_image)


    def apply_sam_prediction(self, image, bbox):
        """Applies SAM prediction to a QImage given a bounding box."""
        if self.sam_model is None:
             print("Error: SAM model not loaded.")
             return None
        try:
            # 1. Convert QImage to NumPy array (BGR, uint8)
            image_np = self.qimage_to_numpy(image)

            # 2. Run SAM prediction
            # SAM/Ultralytics models typically expect RGB format.
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            print(f"Running SAM prediction on image of shape {image_rgb.shape}...")
            results = self.sam_model(image_rgb, bboxes=[bbox]) # Pass RGB array

            if not results:
                print("SAM prediction returned no results.")
                return None

            result = results[0] # Get the first result object

            if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
                print("SAM prediction failed or returned no masks.")
                return None

            # 3. Process Mask
            # Assuming the first mask corresponds to the input bbox
            mask_tensor = result.masks.data[0]
            mask_np = mask_tensor.cpu().numpy() # Move to CPU and convert to numpy
            print(f"Generated mask shape: {mask_np.shape}, dtype: {mask_np.dtype}, range: [{mask_np.min()},{mask_np.max()}]")

            # 4. Convert mask to polygon(s)
            contours = self.mask_to_polygon(mask_np)
            if not contours:
                print("No valid contours found from SAM mask.")
                return None

            # 5. Get confidence score (if available)
            confidence = 0.0
            if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) > 0:
                 confidence = float(result.boxes.conf[0])

            # 6. Format prediction output
            # Return the first/largest contour for simplicity with SAM bbox input
            prediction = {
                "segmentation": contours[0],
                "score": confidence
            }
            print(f"SAM Prediction successful: Score={confidence:.3f}, Polygon points={len(contours[0])//2}")
            return prediction

        except Exception as e:
            print(f"Error during SAM prediction pipeline: {str(e)}")
            traceback.print_exc()
            return None

    def mask_to_polygon(self, mask):
        """Converts a binary mask to polygon coordinates using OpenCV."""
        if mask is None or mask.size == 0:
             return []
        # Ensure mask is binary (0 or 255) and uint8
        # Threshold might depend on mask output range (often 0-1 float)
        threshold = 0.5
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        if contours:
             # Optionally sort by area and take the largest, or handle all
             # contours = sorted(contours, key=cv2.contourArea, reverse=True)
             for contour in contours:
                 if cv2.contourArea(contour) > 10: # Filter small noise contours
                     # Simplify contour slightly? Optional. Might reduce points significantly.
                     # epsilon = 0.001 * cv2.arcLength(contour, True)
                     # approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                     # polygon = approx_contour.flatten().tolist()

                     polygon = contour.flatten().tolist() # Use raw contour points

                     # Polygon needs at least 3 points (6 coordinates)
                     if len(polygon) >= 6:
                         polygons.append(polygon)

        # print(f"Generated {len(polygons)} valid polygons from mask")
        return polygons
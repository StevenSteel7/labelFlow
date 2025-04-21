# utils.py (Migrated to PyQt6 - No changes needed)
"""
Utility functions for the Image Annotator application.

This module contains helper functions used across the application.

@AutoTAG
Mr. Steven Moses # Original Author Note - Keep or Update as needed
"""

import numpy as np
import warnings # Import warnings

def calculate_area(annotation):
    """
    Calculates the area of an annotation (polygon or bounding box).

    Args:
        annotation (dict): An annotation dictionary containing either
                           'segmentation' (list of flat coordinates) or 'bbox' [x,y,w,h].

    Returns:
        float: The calculated area, or 0 if invalid.
    """
    if "segmentation" in annotation and annotation["segmentation"]:
        segmentation = annotation["segmentation"]
        # Handle potential nested list from COCO [[poly]]
        if isinstance(segmentation[0], list):
            segmentation = segmentation[0]

        if len(segmentation) >= 6: # Need at least 3 points
            try:
                # Polygon area using Shoelace formula
                x = np.array(segmentation[0::2])
                y = np.array(segmentation[1::2])
                # Ensure closing point isn't duplicated for calculation
                if np.allclose(x[0], x[-1]) and np.allclose(y[0], y[-1]):
                    x = x[:-1]
                    y = y[:-1]
                if len(x) < 3: return 0.0 # Check again after removing closing point
                return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not calculate polygon area: {e}")
                return 0.0
        else:
             return 0.0 # Not enough points for a polygon

    elif "bbox" in annotation and annotation["bbox"] and len(annotation["bbox"]) == 4:
        try:
            # Rectangle area
            # Ensure w and h are non-negative
            w = float(annotation["bbox"][2])
            h = float(annotation["bbox"][3])
            return max(0.0, w) * max(0.0, h)
        except (ValueError, IndexError, TypeError) as e:
            print(f"Warning: Could not calculate bbox area: {e}")
            return 0.0
    return 0.0 # Return 0 if neither valid segmentation nor bbox found

def calculate_bbox(segmentation):
    """
    Calculates the bounding box [x_min, y_min, width, height] for a given segmentation.

    Args:
        segmentation (list): A flat list of polygon coordinates [x1, y1, x2, y2,...].

    Returns:
        list: A list representing the bounding box [x_min, y_min, width, height],
              or [0, 0, 0, 0] if segmentation is invalid.
    """
    if not segmentation or len(segmentation) < 2:
        return [0.0, 0.0, 0.0, 0.0] # Return default/invalid bbox

    try:
        # Extract coordinates safely
        x_coordinates = np.array(segmentation[0::2], dtype=float)
        y_coordinates = np.array(segmentation[1::2], dtype=float)

        if x_coordinates.size == 0 or y_coordinates.size == 0:
            return [0.0, 0.0, 0.0, 0.0]

        x_min = np.min(x_coordinates)
        y_min = np.min(y_coordinates)
        x_max = np.max(x_coordinates)
        y_max = np.max(y_coordinates)

        width = x_max - x_min
        height = y_max - y_min

        # Ensure width and height are non-negative
        width = max(0.0, width)
        height = max(0.0, height)

        return [x_min, y_min, width, height]
    except (ValueError, IndexError, TypeError) as e:
        print(f"Warning: Could not calculate bounding box: {e}")
        return [0.0, 0.0, 0.0, 0.0] # Return default/invalid bbox on error

def normalize_image(image_array):
    """
    Normalize image array to 8-bit range (0-255).
    Handles potential division by zero for flat images.
    """
    if image_array is None: return None
    if image_array.dtype == np.uint8:
        return image_array # Already 8-bit

    try:
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        range_val = max_val - min_val

        if range_val == 0:
            # Handle flat image - return array of zeros or mean value?
            # Returning zeros is common for visualization normalization.
            # If preserving original value is important, adjust this.
            return np.zeros_like(image_array, dtype=np.uint8)

        # Normalize using float64 for precision, then scale and cast to uint8
        normalized = (image_array.astype(np.float64) - min_val) / range_val
        # Clip before casting just in case of floating point inaccuracies near 0 or 1
        scaled = np.clip(normalized * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    except (ValueError, TypeError, FloatingPointError) as e:
        # Catch potential errors during numpy operations
        print(f"Warning: Could not normalize image: {e}")
        # Fallback: Attempt simple cast if possible, otherwise return None or raise
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore cast warnings
             try:
                  # Try clipping and casting as a last resort
                  return np.clip(image_array, 0, 255).astype(np.uint8)
             except:
                  return None # Return None if all normalization fails
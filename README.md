# labelFlow

A comprehensive desktop application built with PyQt5 for image annotation, featuring manual and AI-assisted tools (SAM, YOLO), support for various image formats (including multi-slice scientific formats), project management, and dataset utilities.

Developed by Mr. Steven Moses.

## Key Features

*   **Manual Annotation:** Polygon, Rectangle, Paint Brush, Eraser tools.
*   **AI-Assisted Segmentation:** Integrated Segment Anything Model (SAM via Ultralytics) using Magic Wand.
*   **YOLO Integration:** Prepare data, train, and predict using YOLO models (via Ultralytics).
*   **Multi-format Support:** Handles standard images (PNG, JPG, BMP) and multi-slice scientific formats (TIFF, CZI, DICOM).
*   **Project Management:** Create, open, save, and search annotation projects (`.iap` format).
*   **Annotation Management:** Class creation/editing, annotation list, statistics, merging.
*   **Import/Export:** Supports COCO JSON, YOLO (v4, v5+), Pascal VOC, Labeled Images, Semantic Labels.
*   **Dataset Utilities:** Tools for splitting datasets, patching images, augmenting images, slice registration, stack interpolation.
*   **UI Customization:** Includes Dark Mode and font size options.

## Installation

**1. Prerequisites:**

*   **Python:** Version 3.12 or higher is recommended.
*   **Package Manager:** `uv` (recommended, faster) or `pip`.
*   **(Optional but Recommended)** A virtual environment manager (like Python's built-in `venv`).

   ```bash
   # Example using venv
   python -m venv .venv
   # Activate on Windows: .\.venv\Scripts\activate
   # Activate on macOS/Linux: source .venv/bin/activate


2. Install PyTorch (REQUIRED FIRST STEP):

labelFlow requires PyTorch for its AI features (SAM and YOLO). You must install PyTorch before installing labelFlow.

Go to the official PyTorch website.

Select your System (Linux, Mac, Windows), Package manager (uv or pip), Compute Platform (CPU or specific CUDA version), and Python version.

Copy and run the exact command provided by the website.

Example for CPU-only:
```bash
# Using uv
uv pip install torch torchvision torchaudio

# Using pip
  pip install torch torchvision torchaudio
  ```
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Example for CUDA 12.6 (Verify command on PyTorch website!):
```bash
# Using uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Using pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Ensure PyTorch installs successfully before proceeding to the next step.

3. Install labelFlow:

Once PyTorch is correctly installed, you can install labelFlow directly from GitHub:

# Using uv
uv pip install git+https://github.com/StevenSteel7/labelFlow.git

# Using pip
pip install git+https://github.com/StevenSteel7/labelFlow.git
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This command will download the code and install labelFlow along with its other dependencies (PyQt5, Ultralytics, tifffile, etc.) listed in pyproject.toml.

(Alternative for Development): If you have cloned the repository (git clone https://github.com/StevenSteel7/labelFlow.git), you can install it in editable mode from the project's root directory:

# Navigate to the cloned directory first
cd labelFlow

# Using uv
uv pip install -e .

# Using pip
pip install -e .
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Usage

After successful installation, activate your virtual environment (if you used one) and run the application from your terminal:

labelflow
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
License

This project is licensed under the [Your License Name] License - see the LICENSE file for details. (Please add a LICENSE file and specify the license name here, e.g., MIT License)

Author

Mr. Steven Moses

Save this content as `README.md` in the root of your `labelFlow` project directory on GitHub.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

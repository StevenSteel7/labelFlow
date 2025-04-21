# help_window.py (Migrated to PyQt6)

# PyQt6 Imports
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QApplication # Added QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont # QFont remains in QtGui

# Local Imports (Assuming these are PyQt6 compatible)
from .soft_dark_stylesheet import soft_dark_stylesheet
from .default_stylesheet import default_stylesheet

class HelpWindow(QDialog):
    def __init__(self, dark_mode=False, font_size=10, parent=None): # Added parent=None
        super().__init__(parent) # Pass parent to superclass
        self.setWindowTitle("Help Guide - Image Annotator")
        self.setModal(False)  # Keep it non-modal
        # Use Qt.WindowType enum
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setGeometry(100, 100, 800, 600) # Initial size

        layout = QVBoxLayout(self) # Set layout on self directly
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        layout.addWidget(self.text_browser)
        # No need to call self.setLayout(layout) when passing self to layout constructor

        self.dark_mode = dark_mode # Store mode
        self.font_size = font_size # Store font size

        self.apply_styles() # Apply styles and font
        self.load_help_content()

    def apply_styles(self):
        """Applies the stylesheet and font size."""
        # Base stylesheet
        base_style = soft_dark_stylesheet if self.dark_mode else default_stylesheet
        # Font size override for the entire dialog and its children
        font_style = f"QWidget {{ font-size: {self.font_size}pt; }}"
        # Combine styles
        self.setStyleSheet(base_style + "\n" + font_style)

        # --- Optional: Set base font specifically for text browser ---
        # This ensures the HTML content uses the base size, but internal
        # HTML tags like <h1> can still override it.
        # font = self.text_browser.font()
        # font.setPointSize(self.font_size)
        # self.text_browser.setFont(font)
        # --- End Optional ---


    def load_help_content(self):
        """Loads the HTML help content into the text browser."""
        # Consider loading from an external HTML file for easier maintenance
        # e.g., with open("help.html", "r") as f: help_text = f.read()
        help_text = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ /* Style body if needed, inherits from QWidget style */
                    /* font-size: {self.font_size}pt; /* Set base size for body */
                }}
                h1 {{ font-size: {self.font_size + 6}pt; font-weight: bold; margin-bottom: 10px; }}
                h2 {{ font-size: {self.font_size + 4}pt; font-weight: bold; margin-top: 15px; margin-bottom: 8px; }}
                h3 {{ font-size: {self.font_size + 2}pt; font-weight: bold; margin-top: 12px; margin-bottom: 5px; }}
                p {{ margin-bottom: 8px; line-height: 1.4; }}
                ul, ol {{ margin-left: 20px; margin-bottom: 8px; }}
                li {{ margin-bottom: 4px; }}
                strong {{ font-weight: bold; }}
                /* Add specific link styling if needed, otherwise inherits QWidget color */
                /* a {{ color: #5cacee; text-decoration: none; }} */
                /* a:hover {{ text-decoration: underline; }} */
            </style>
        </head>
        <body>
            <h1>Image Annotator Help Guide</h1>

            <h2>Overview</h2>
            <p>Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in various formats, including COCO-style JSON, YOLO v8, and Pascal VOC. Annotations can be defined using manual tools like the polygon tool or in a semi-automated way with the assistance of the Segment Anything Model (SAM-2) pre-trained model. The tool supports multi-dimensional images such as TIFF stacks and CZI files and provides dark mode and adjustable application font sizes for enhanced GUI experience.</p>

            <h2>Keyboard Shortcuts</h2>
            <ul>
                <li><strong>Shift + Click (in Edit Mode):</strong> Delete a polygon vertex</li>
                <li><strong>Double Click (on polygon):</strong> Enter polygon edit mode</li>
                <li><strong>Enter (in Edit Mode):</strong> Exit polygon edit mode, save changes</li>
                <li><strong>Enter (drawing polygon):</strong> Finish polygon</li>
                <li><strong>Enter (SAM mode):</strong> Accept SAM prediction</li>
                <li><strong>Enter (Paint/Eraser mode):</strong> Commit Paint/Eraser changes</li>
                <li><strong>Esc:</strong> Cancel current action (drawing polygon/rectangle, SAM box), exit edit mode, discard SAM prediction, discard Paint/Eraser changes</li>
                <li><strong>Ctrl + Wheel:</strong> Zoom in/out on image</li>
                <li><strong>Ctrl + Left Mouse Drag:</strong> Pan image</li>
                <li><strong>+/- Keys (Paint/Eraser mode):</strong> Decrease/Increase brush/eraser size</li>
                <li><strong>Ctrl + N:</strong> New Project</li>
                <li><strong>Ctrl + O:</strong> Open Project</li>
                <li><strong>Ctrl + S:</strong> Save Project</li>
                <li><strong>Ctrl + Shift + S:</strong> Save Project As...</li>
                <li><strong>Ctrl + W:</strong> Close Project</li>
                <li><strong>Ctrl + I:</strong> Project Details</li>
                <li><strong>Ctrl + F:</strong> Search Projects</li>
                <li><strong>Ctrl + Alt + S:</strong> Annotation Statistics</li>
                <li><strong>Ctrl + D:</strong> Toggle Dark Mode</li>
                <li><strong>Delete Key (List focused):</strong> Delete selected class / annotation / image</li>
                <li><strong>Up/Down Arrow Keys (Slice list focused):</strong> Navigate slices</li>
                <li><strong>F1:</strong> Show Help (This window)</li>
                 <li><strong>F2:</strong> Launch Snake Game</li>
            </ul>


            <h2>Key Features</h2>
            <ul>
                <li>Manual annotations with polygons, rectangles, paint brush, and eraser</li>
                <li>Semi-automated segmentation with SAM-2 assistance (requires model selection)</li>
                <li>Save and load projects (.iap format)</li>
                <li>Import existing COCO JSON and YOLO annotations with images</li>
                <li>Export annotations to various formats (COCO, YOLOv4, YOLOv5+, Labeled Images, Semantic Labels, Pascal VOC)</li>
                <li>Handle multi-dimensional images (TIFF stacks, CZI) with dimension assignment</li>
                <li>Support for standard image formats (PNG, JPG, BMP)</li>
                <li>Annotation editing (move/add/delete polygon vertices)</li>
                <li>Merge connected annotations of the same class</li>
                <li>Class management (add, rename, delete, change color, toggle visibility)</li>
                <li>Zoom and pan functionality</li>
                <li>Dark mode and adjustable font size</li>
                <li>YOLOv8 integration (beta): dataset preparation, training, prediction</li>
                <li>Additional Tools: Dataset Splitter, COCO Combiner, Stack->Slices, Patcher, Augmenter, Slice Registration, Stack Interpolator, DICOM Converter</li>
            </ul>

            <h2>Getting Started</h2>
            <h3>Starting a New Project</h3>
            <ol>
                <li>Go to <b>Project > New Project</b> (Ctrl+N). Choose a location and name for your project file (.iap).</li>
                <li>Click <b>Add New Images</b> to import images. You can select multiple files, including multi-dimensional TIFF/CZI.</li>
                <li>If multi-dimensional images are added, you'll be prompted to assign dimensions (e.g., T, Z, C, H, W). Ensure 'H' and 'W' are assigned correctly.</li>
                <li>Click <b>Add Classes</b> to define object categories (e.g., "Cell", "Nucleus").</li>
                <li>Select an image from the list on the right.</li>
                <li>Select a class from the list on the left.</li>
                <li>Choose an annotation tool (Polygon, Rectangle, Paint, Eraser, or SAM-Assisted) and start annotating.</li>
                <li>Save your project frequently (Ctrl+S).</li>
            </ol>

            <h3>Opening an Existing Project</h3>
            <ol>
                <li>Go to <b>Project > Open Project</b> (Ctrl+O) and select your .iap file.</li>
                <li>The application will load images from the 'images' subdirectory within the project folder.</li>
                <li>If images are missing, you'll be prompted to locate them. Located images are copied into the project's 'images' folder.</li>
            </ol>

            <h3>Importing Existing Annotations</h3>
            <ol>
                <li>Go to <b>Project > New Project</b> or <b>Open Project</b> first.</li>
                <li>Click <b>Import Annotations with Images</b> in the sidebar.</li>
                <li>Select the format (COCO JSON or YOLO).</li>
                <li>Select the annotation file (JSON or YAML).</li>
                <li>Ensure the corresponding image files are located correctly relative to the annotation file (e.g., in an 'images' subdirectory for COCO, or as specified in the YOLO YAML).</li>
                <li>Annotations and images will be added to the current project.</li>
            </ol>

            <h2>Annotation Process</h2>
            <ol>
                <li><strong>Select Image/Slice:</strong> Use the 'Images' or 'Slices' list on the right.</li>
                <li><strong>Select Class:</strong> Click a class name in the 'Classes' list.</li>
                <li><strong>Choose Tool:</strong> Click a tool button (Polygon, Rectangle, Paint, Eraser, SAM-Assisted).</li>
                <li><strong>Annotate:</strong>
                    <ul>
                        <li><b>Polygon:</b> Left-click to place vertices around the object. Double-click near the start point or press Enter to finish. Press Esc to cancel.</li>
                        <li><b>Rectangle:</b> Left-click and drag to draw a box. Release mouse to finish.</li>
                        <li><b>Paint Brush:</b> Left-click and drag to paint areas. Press Enter to commit the painted mask as polygon(s). Press Esc to discard. Use +/- keys or Ctrl+Wheel to change size.</li>
                        <li><b>Eraser:</b> Left-click and drag to erase parts of existing annotations. Press Enter to commit changes. Press Esc to discard. Use +/- keys or Ctrl+Wheel to change size.</li>
                        <li><b>SAM-Assisted:</b>
                            <ol>
                                <li>Select a SAM model from the dropdown (downloads on first use). Smaller models (tiny, small) are faster.</li>
                                <li>The "SAM-Assisted" button becomes enabled and checked.</li>
                                <li>Left-click and drag to draw a bounding box around the object of interest.</li>
                                <li>Release the mouse. A preview of the segmented mask (highest confidence) will appear in orange.</li>
                                <li>If satisfied, press Enter to accept the annotation.</li>
                                <li>If not satisfied, press Esc to clear the preview and try drawing a different box.</li>
                                <li>Click another tool button or uncheck "SAM-Assisted" to deactivate.</li>
                            </ol>
                        </li>
                    </ul>
                </li>
                 <li><strong>Edit Polygon:</strong> Double-click inside an existing polygon annotation. Handles appear. Drag handles to move vertices. Click on an edge to add a vertex. Shift+Click a vertex to delete it. Press Enter or Esc to finish editing.</li>
                 <li><strong>Delete Annotation:</strong> Select annotation(s) in the 'Annotations' list and press the Delete key or click the 'Delete' button.</li>
                 <li><strong>Merge Annotations:</strong> Select two or more connected polygon annotations of the *same class* in the list, then click the 'Merge' button.</li>
                 <li><strong>Change Annotation Class:</strong> Select annotation(s) in the list, click 'Change Class', and choose the new class.</li>
            </ol>

            <h2>Exporting Annotations</h2>
            <ol>
                <li>Select the desired export format from the dropdown in the sidebar.</li>
                <li>Click <b>Export Annotations</b>.</li>
                <li>Choose the export location (file for COCO JSON, directory for others) and confirm.</li>
                <li>The annotations and necessary image files (if applicable to the format) will be saved.</li>
            </ol>

            <h2>Navigation and Viewing</h2>
            <ul>
                <li><strong>Zoom:</strong> Use the slider, or hold Ctrl + Mouse Wheel.</li>
                <li><strong>Pan:</strong> Hold Ctrl + Left Mouse Button Drag.</li>
                <li><strong>Switch Images:</strong> Click image name in the 'Images' list.</li>
                <li><strong>Navigate Slices:</strong> Use 'Slices' list or Up/Down arrow keys when list has focus.</li>
                <li><strong>Class Visibility:</strong> Check/uncheck boxes next to class names to show/hide annotations for that class.</li>
            </ul>

            <h2>Tools Menu</h2>
            <p>The <b>Tools</b> menu provides utilities for dataset management and image processing:</p>
            <ul>
                <li><strong>Annotation Statistics:</strong> View counts and distribution of annotations.</li>
                <li><strong>COCO JSON Combiner:</strong> Merge multiple COCO annotation files.</li>
                <li><strong>Dataset Splitter:</strong> Split images (and optionally COCO/YOLO annotations) into train/val/test sets.</li>
                <li><strong>Stack to Slices:</strong> Convert TIFF/CZI stacks into individual 2D PNG slices based on assigned dimensions.</li>
                <li><strong>Image Patcher:</strong> Divide large images into smaller overlapping or non-overlapping patches.</li>
                <li><strong>Image Augmenter:</strong> Apply random transformations (rotate, flip, blur, etc.) to images (and optionally COCO annotations).</li>
                <li><strong>Slice Registration:</strong> Align slices in an image stack using various transformation methods.</li>
                <li><strong>Stack Interpolator:</strong> Resample an image stack to new voxel/pixel spacing (e.g., create isotropic volumes).</li>
                <li><strong>DICOM Converter:</strong> Convert DICOM files (.dcm) to TIFF format, preserving metadata in a JSON file.</li>
            </ul>

            <h2>Troubleshooting & Tips</h2>
            <ul>
                 <li>Ensure multi-dimensional images (TIFF, CZI) have correct dimensions ('H', 'W') assigned via the initial dialog or the 'Redefine Dimensions' option (right-click image).</li>
                 <li>SAM works best on clear objects. For complex or low-contrast areas, manual tools might be needed.</li>
                 <li>Save your project (.iap file) regularly.</li>
                 <li>When importing COCO/YOLO, ensure the image files are correctly located relative to the annotation file.</li>
                 <li>Check console output for warnings or errors during operations like import, export, or training.</li>
                 <li>For YOLO training, ensure the dataset structure matches the format requirements (v4 or v5+).</li>
            </ul>

            <h2>Getting Help</h2>
            <p>If you encounter issues or have suggestions, please consult the project documentation or report an issue on the relevant GitHub repository.</p>
        </body>
        </html>
        """
        self.text_browser.setHtml(help_text)

    def show_centered(self, parent):
        """Shows the dialog centered on the parent window."""
        if parent:
            parent_geo = parent.geometry()
            # Ensure self has valid geometry before calculating center
            if self.rect().isValid():
                 self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents() # Ensure window displays properly

# Function to show the help window (can be called from main app)
def show_help_window(parent, dark_mode, font_size):
    dialog = HelpWindow(dark_mode=dark_mode, font_size=font_size, parent=parent)
    dialog.show_centered(parent)
    # No need to return the dialog if it's non-modal and manages its own lifecycle
    # return dialog
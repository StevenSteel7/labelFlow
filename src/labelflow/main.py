# main.py (Migrated to PyQt6 - Further Corrected High DPI Handling)
"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import sys
import os
# PyQt6 Imports
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt # Import Qt is needed for RoundingPolicy
# Local Import (assuming it's PyQt6 compatible)
from .annotator_window import ImageAnnotator

# Linux QT_QPA_PLATFORM_PLUGIN_PATH handling (keep commented unless needed)
# ... (code omitted for brevity) ...

def main():
    """
    Main function to run the Image Annotator application.
    """

    # --- High DPI Scaling Settings (Set *before* QApplication) ---
    # Set rounding policy before creating the app object
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough) # Use enum

    # AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps attributes are generally
    # deprecated/removed in Qt6. High DPI scaling is usually enabled by default
    # or controlled via environment variables (e.g., QT_ENABLE_HIGHDPI_SCALING=1).
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling) # REMOVED
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps) # REMOVED

    # Create QApplication instance *after* setting attributes/policies
    app = QApplication(sys.argv)

    # Create and show the main window
    window = ImageAnnotator()
    window.show()

    # Start the application event loop
    sys.exit(app.exec()) # Use exec() in PyQt6

if __name__ == "__main__":
    main()
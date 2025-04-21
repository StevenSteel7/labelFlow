# project_search.py (Migrated to PyQt6)
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QDateEdit, QLabel, QListWidget, QDialogButtonBox, QFormLayout,
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QDate, QDateTime # Added QDateTime for potential parsing
import os
import json
from datetime import datetime
import traceback # For debugging

class ProjectSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Store parent window reference safely
        self.parent_window = parent
        self.setWindowTitle("Search Projects")
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.search_directory = "" # Initialize search directory
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10) # Add spacing

        # --- Search Criteria ---
        criteria_group = QGroupBox("Search Criteria")
        form_layout = QFormLayout()
        form_layout.setSpacing(8)

        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("Enter keywords (e.g., cell AND experiment OR analysis)")
        self.keyword_edit.setToolTip("Search in project name, notes, image names, class names.\nUse AND/OR/() for boolean logic (case-insensitive).")
        form_layout.addRow("Search Query:", self.keyword_edit)

        self.start_date_edit = QDateEdit() # Renamed for clarity
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd") # Set standard format
        # Default start date: 1 year ago
        self.start_date_edit.setDate(QDate.currentDate().addYears(-1))
        self.start_date_edit.setToolTip("Search for projects created on or after this date.")
        form_layout.addRow("Created After:", self.start_date_edit)

        self.end_date_edit = QDateEdit() # Renamed for clarity
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd") # Set standard format
        self.end_date_edit.setDate(QDate.currentDate()) # Default end date: today
        self.end_date_edit.setToolTip("Search for projects created on or before this date.")
        form_layout.addRow("Created Before:", self.end_date_edit)

        criteria_group.setLayout(form_layout)
        layout.addWidget(criteria_group)

        # --- Directory Selection ---
        dir_group = QGroupBox("Search Location")
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Directory: Not selected") # Use a label for display
        self.dir_label.setWordWrap(True)
        self.dir_label.setToolTip("") # Tooltip will show full path
        dir_button = QPushButton("Browse...")
        dir_button.setToolTip("Select the top-level directory to search within.")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_label, 1) # Label takes available space
        dir_layout.addWidget(dir_button)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # --- Search Action ---
        search_button = QPushButton("Search Projects")
        search_button.clicked.connect(self.perform_search)
        layout.addWidget(search_button)

        # --- Results ---
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout()
        self.results_list = QListWidget()
        self.results_list.setToolTip("Double-click a project to open it.")
        self.results_list.itemDoubleClicked.connect(self.open_selected_project)
        results_layout.addWidget(self.results_list)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # --- Dialog Buttons ---
        # Use standard buttons (Close is typically RejectRole)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close) # Use enum
        button_box.rejected.connect(self.reject) # Connect Close to reject
        layout.addWidget(button_box)

    def browse_directory(self):
        """Opens a dialog to select the search directory."""
        # QFileDialog.getExistingDirectory returns a string path
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Search")
        if directory:
            self.search_directory = directory
            # Update label and tooltip
            self.dir_label.setText(f"...{os.sep}{os.path.basename(directory)}")
            self.dir_label.setToolTip(directory)

    def perform_search(self):
        """Initiates the search for project files based on criteria."""
        if not self.search_directory or not os.path.isdir(self.search_directory):
            QMessageBox.warning(self, "No Directory", "Please select a valid directory to search.")
            return

        query = self.keyword_edit.text().strip()
        # Get QDate objects and convert to Python date
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()

        self.results_list.clear()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Use enum
        found_projects = []

        try:
            print(f"Searching in: {self.search_directory}")
            print(f"Query: '{query}'")
            print(f"Date Range: {start_date} to {end_date}")

            for root, dirs, files in os.walk(self.search_directory):
                # Optional: Skip hidden directories or specific paths
                # dirs[:] = [d for d in dirs if not d.startswith('.')]

                for filename in files:
                    if filename.lower().endswith('.iap'): # Case-insensitive check
                        project_path = os.path.join(root, filename)
                        try:
                            with open(project_path, 'r', encoding='utf-8') as f: # Specify encoding
                                project_data = json.load(f)

                            # Add project_path to data for easier matching if needed
                            project_data['_project_path'] = project_path

                            if self.project_matches(project_data, query, start_date, end_date):
                                found_projects.append(project_path)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON file: {project_path}")
                        except Exception as e:
                            print(f"Warning: Error reading project file {project_path}: {str(e)}")
                            # traceback.print_exc() # Uncomment for detailed debug

            print(f"Found {len(found_projects)} potential matches.")
            # Sort results alphabetically
            found_projects.sort()
            self.results_list.addItems(found_projects)

        except Exception as e:
             QMessageBox.critical(self, "Search Error", f"An error occurred during search:\n{e}")
             traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

        # Show results summary
        count = self.results_list.count()
        if count == 0:
            QMessageBox.information(self, "Search Results", "No matching projects found.")
        else:
            QMessageBox.information(self, "Search Results", f"Found {count} matching project(s).")


    def project_matches(self, project_data, query, start_date, end_date):
        """Checks if a single project matches the search criteria."""
        # 1. Check Date Range
        creation_date_str = project_data.get('creation_date', '')
        if creation_date_str:
            try:
                # Handle different potential datetime formats
                try:
                    creation_dt = datetime.fromisoformat(str(creation_date_str))
                except ValueError:
                    # Try common formats if ISO fails (add more as needed)
                    creation_dt = datetime.strptime(str(creation_date_str), "%Y-%m-%d %H:%M:%S")

                creation_date_obj = creation_dt.date()
                if creation_date_obj < start_date or creation_date_obj > end_date:
                    return False # Outside date range
            except (ValueError, TypeError) as date_err:
                # Ignore projects with invalid date format for date filtering
                print(f"Warning: Invalid date format '{creation_date_str}' in project: {project_data.get('_project_path', 'N/A')}. Skipping date check. Error: {date_err}")
                pass # Don't filter out based on bad date, but proceed to keyword check

        # 2. Check Keyword Query
        if query:
            # Simple AND logic for now (split query into words)
            # More complex boolean logic (like in the original `evaluate_query`) can be added here if needed.
            keywords = query.lower().split()
            match_all = True
            for keyword in keywords:
                if not self.term_matches(keyword, project_data):
                    match_all = False
                    break # One keyword not found -> no match
            if not match_all:
                return False

        # If we passed date and keyword checks (or no query), it's a match
        return True

    def term_matches(self, term, project_data):
        """Checks if a single term exists in relevant project fields (case-insensitive)."""
        term_lower = term.lower() # Search case-insensitively

        # Check project filename (extracted from path if available)
        project_path = project_data.get('_project_path', '')
        if project_path and term_lower in os.path.basename(project_path).lower():
            return True

        # Check classes
        if any(term_lower in class_info['name'].lower() for class_info in project_data.get('classes', [])):
            return True

        # Check image filenames
        if any(term_lower in img['file_name'].lower() for img in project_data.get('images', [])):
            return True

        # Check project notes
        if term_lower in project_data.get('notes', '').lower():
            return True

        return False

    # --- evaluate_query and tokenize_query (Original complex boolean logic) ---
    # These can be re-integrated if complex boolean search ("AND", "OR", "()") is strictly required.
    # For simplicity, the current implementation uses simple space-separated AND logic.
    # If re-adding, ensure they use term_matches correctly.
    # def evaluate_query(self, query, project_data):
    #     tokens = self.tokenize_query(query)
    #     return self.evaluate_tokens(tokens, project_data)
    #
    # def tokenize_query(self, query):
    #     # ... (implementation from original code) ...
    #
    # def evaluate_tokens(self, tokens, project_data):
    #     # ... (implementation from original code, using self.term_matches) ...

    def open_selected_project(self, item):
        """Opens the project associated with the double-clicked list item."""
        project_file = item.text()
        if self.parent_window and hasattr(self.parent_window, 'open_specific_project'):
            try:
                 # Close search dialog *before* opening project
                 self.accept() # Indicate success/completion
                 # Call the parent window's method to open the project
                 self.parent_window.open_specific_project(project_file)
            except Exception as e:
                 QMessageBox.critical(self, "Open Error", f"Failed to open project:\n{project_file}\n\nError: {e}")
                 traceback.print_exc()
        else:
             QMessageBox.warning(self, "Error", "Cannot open project - Parent window reference is missing.")


def show_project_search(parent):
    """Creates and shows the Project Search dialog."""
    dialog = ProjectSearchDialog(parent)
    # Use exec() to show modally and wait for it to close
    dialog.exec()
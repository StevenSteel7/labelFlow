# snake_game.py (Migrated to PyQt6)
import sys
import random
# PyQt6 Imports
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6.QtGui import QPainter, QColor, QScreen # QDesktopWidget is deprecated, use QScreen
from PyQt6.QtCore import Qt, QTimer

class SnakeGame(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Secret Snake Game (PyQt6)')
        self.setFixedSize(600, 400)  # Increased size
        self.center() # Center the window on creation

        self.cell_size = 10 # Define cell size
        self.board_width_cells = self.width() // self.cell_size
        self.board_height_cells = self.height() // self.cell_size

        # Initial snake position (using cell indices then multiplying by cell_size)
        start_x = self.board_width_cells // 2
        start_y = self.board_height_cells // 2
        self.snake = [
            (start_x * self.cell_size, start_y * self.cell_size),
            ((start_x - 1) * self.cell_size, start_y * self.cell_size),
            ((start_x - 2) * self.cell_size, start_y * self.cell_size)
        ]
        self.direction = Qt.Key.Key_Right # Use Qt.Key enum for direction tracking
        self.food = self.place_food()
        self.score = 0
        self.is_paused = False
        self.game_over_flag = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)
        self.timer.start(100) # Game speed (milliseconds)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # Use enum
        self.show()

    def center(self):
        """Centers the widget on the primary screen."""
        if hasattr(self, 'screen'): # Check if screen() method exists (newer PyQt6)
            center_point = self.screen().availableGeometry().center()
        else: # Fallback for older PyQt6 or cases where screen isn't attached yet
             try:
                  # Get primary screen directly from QApplication
                  primary_screen = QApplication.primaryScreen()
                  if primary_screen:
                      center_point = primary_screen.availableGeometry().center()
                  else: # Fallback if no primary screen found
                      print("Warning: Could not determine primary screen center. Placing window at default position.")
                      return
             except Exception: # Catch potential errors getting screen info early
                  print("Warning: Could not determine screen center. Placing window at default position.")
                  return

        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def paintEvent(self, event):
        """Handles painting the game elements."""
        painter = QPainter(self)
        # Antialiasing might not be needed for simple rectangles, but keep if preferred
        # painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # --- Draw Background (Optional) ---
        # painter.fillRect(self.rect(), QColor(200, 200, 200)) # Light gray background

        # --- Draw Snake ---
        # Head different color?
        # painter.setBrush(QColor(0, 200, 0)) # Darker Green Head
        # head = self.snake[0]
        # painter.drawRect(head[0], head[1], self.cell_size, self.cell_size)

        # Body
        painter.setBrush(QColor(0, 255, 0)) # Green body
        for segment in self.snake: # Draw all segments same color for simplicity now
            painter.drawRect(segment[0], segment[1], self.cell_size, self.cell_size)

        # --- Draw Food ---
        painter.setBrush(QColor(255, 0, 0)) # Red food
        painter.drawRect(self.food[0], self.food[1], self.cell_size, self.cell_size)

        # --- Draw Score ---
        painter.setPen(QColor(0, 0, 0)) # Black text
        font = painter.font()
        font.setPointSize(12) # Make score slightly larger
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 20, f"Score: {self.score}")

        # --- Draw Pause/Game Over Message ---
        if self.is_paused:
             painter.setPen(QColor(100, 100, 100))
             font.setPointSize(20)
             painter.setFont(font)
             painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "PAUSED (P)") # Use enum
        elif self.game_over_flag:
             painter.setPen(QColor(200, 0, 0))
             font.setPointSize(20)
             painter.setFont(font)
             painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"GAME OVER! Score: {self.score}\n(Press Enter to Restart)") # Use enum

    def keyPressEvent(self, event):
        """Handles keyboard input for snake control and game actions."""
        key = event.key()

        # Restart game if game over
        if self.game_over_flag and (key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter):
            self.restart_game()
            return

        # Pause game
        if key == Qt.Key.Key_P:
            self.toggle_pause()
            return

        # If paused or game over, ignore direction keys
        if self.is_paused or self.game_over_flag:
            return

        # --- Direction Handling ---
        # Prevent immediate reversal
        current_direction = self.direction
        new_direction = current_direction # Default to current

        if key == Qt.Key.Key_Left and current_direction != Qt.Key.Key_Right:
            new_direction = Qt.Key.Key_Left
        elif key == Qt.Key.Key_Right and current_direction != Qt.Key.Key_Left:
            new_direction = Qt.Key.Key_Right
        elif key == Qt.Key.Key_Up and current_direction != Qt.Key.Key_Down:
            new_direction = Qt.Key.Key_Up
        elif key == Qt.Key.Key_Down and current_direction != Qt.Key.Key_Up:
            new_direction = Qt.Key.Key_Down
        elif key == Qt.Key.Key_Escape:
            self.close() # Allow closing anytime
            return

        self.direction = new_direction # Update direction for the next game tick

    def toggle_pause(self):
        """Pauses or resumes the game."""
        if self.game_over_flag: return # Cannot pause if game over
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
        else:
            self.timer.start()
        self.update() # Redraw to show pause message

    def update_game(self):
        """Updates the game state each timer tick."""
        if self.is_paused or self.game_over_flag:
            return # Don't update if paused or game over

        head_x, head_y = self.snake[0]
        new_head_x, new_head_y = head_x, head_y

        # Calculate new head position based on current direction
        if self.direction == Qt.Key.Key_Left:
            new_head_x -= self.cell_size
        elif self.direction == Qt.Key.Key_Right:
            new_head_x += self.cell_size
        elif self.direction == Qt.Key.Key_Up:
            new_head_y -= self.cell_size
        elif self.direction == Qt.Key.Key_Down:
            new_head_y += self.cell_size

        new_head = (new_head_x, new_head_y)

        # --- Check for Collisions ---
        # 1. Wall Collision
        if (new_head_x < 0 or new_head_x >= self.width() or
            new_head_y < 0 or new_head_y >= self.height()):
            self.game_over()
            return

        # 2. Self Collision
        if new_head in self.snake:
            self.game_over()
            return

        # --- Move Snake ---
        self.snake.insert(0, new_head) # Add new head

        # --- Check for Food Collision ---
        if new_head == self.food:
            self.score += 1
            self.food = self.place_food() # Place new food
            # Don't pop tail, snake grows
        else:
            self.snake.pop() # Remove tail segment

        self.update() # Trigger repaint

    def place_food(self):
        """Places food randomly on an empty cell."""
        while True:
            # Calculate food position based on cell grid
            food_x = random.randrange(self.board_width_cells) * self.cell_size
            food_y = random.randrange(self.board_height_cells) * self.cell_size
            food_pos = (food_x, food_y)
            # Ensure food is not placed on the snake
            if food_pos not in self.snake:
                return food_pos

    def game_over(self):
        """Stops the timer and sets the game over flag."""
        self.timer.stop()
        self.game_over_flag = True
        self.update() # Redraw to show game over message
        # Optional: Show message box after a delay or on key press
        # QMessageBox.information(self, "Game Over", f"Your score: {self.score}")

    def restart_game(self):
        """Resets the game state and restarts the timer."""
        self.center() # Re-center in case window moved
        start_x = self.board_width_cells // 2
        start_y = self.board_height_cells // 2
        self.snake = [
            (start_x * self.cell_size, start_y * self.cell_size),
            ((start_x - 1) * self.cell_size, start_y * self.cell_size),
            ((start_x - 2) * self.cell_size, start_y * self.cell_size)
        ]
        self.direction = Qt.Key.Key_Right
        self.food = self.place_food()
        self.score = 0
        self.is_paused = False
        self.game_over_flag = False
        self.timer.start(100) # Restart timer
        self.update() # Initial redraw

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # --- High DPI Scaling (Recommended for PyQt6) ---
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough) # Use enum
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling) # Use enum
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps) # Use enum

    ex = SnakeGame()
    sys.exit(app.exec()) # Use exec()
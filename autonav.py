import cv2
import numpy as np
import threading
import time
import logging
import os
import pickle
from collections import deque
from pathlib import Path
import random

logger = logging.getLogger("AutonomousNavigation")

class MemoryBuffer:
    """Stores experiences for training autonomous navigation"""
    
    def __init__(self, buffer_size=1000, save_dir="navigation_data"):
        self.buffer = deque(maxlen=buffer_size)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        logger.info(f"Memory buffer initialized with size {buffer_size}")
        
    def add_experience(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))
            
    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return random.sample(list(self.buffer), len(self.buffer))
            return random.sample(list(self.buffer), batch_size)
            
    def save(self, filename=None):
        """Save the memory buffer to disk"""
        if filename is None:
            filename = f"memory_buffer_{int(time.time())}.pkl"
        
        filepath = self.save_dir / filename
        with self.lock:
            with open(filepath, 'wb') as f:
                pickle.dump(list(self.buffer), f)
        logger.info(f"Memory buffer saved to {filepath}")
        
    def load(self, filepath):
        """Load a memory buffer from disk"""
        with self.lock:
            # Check file existence and read within the lock to prevent race conditions
            if not os.path.exists(filepath):
                logger.warning(f"File {filepath} does not exist")
                return False
            
            # Read the file within the lock to prevent modification during read
            try:
                with open(filepath, 'rb') as f:
                    loaded_buffer = pickle.load(f)
            except (IOError, OSError, pickle.PickleError) as e:
                logger.error(f"Error reading file {filepath}: {e}")
                return False
            
            # Check if loaded data exceeds maxlen and warn about truncation
            loaded_count = len(loaded_buffer)
            maxlen = self.buffer.maxlen
            
            if loaded_count > maxlen:
                logger.warning(
                    f"Loaded buffer has {loaded_count} experiences, but maxlen is {maxlen}. "
                    f"Truncating to {maxlen} (discarding {loaded_count - maxlen} oldest experiences)."
                )
                # Keep only the most recent experiences
                loaded_buffer = loaded_buffer[-maxlen:]
            
            # Replace the buffer with loaded data
            self.buffer = deque(loaded_buffer, maxlen=maxlen)
            
        logger.info(f"Loaded {len(self.buffer)} experiences from {filepath}")
        return True
        
    def __len__(self):
        return len(self.buffer)


class SimpleObstacleDetector:
    """Detects obstacles using basic image processing"""
    
    def __init__(self):
        self.min_obstacle_area = 500  # Minimum area to consider as obstacle
        logger.info("Simple obstacle detector initialized")
        
    def detect_obstacles(self, frame):
        """
        Detect obstacles in the frame using color thresholding and contour detection
        Returns: List of obstacle bounding boxes [(x, y, w, h), ...]
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to identify potential obstacles
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_obstacle_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    obstacles.append((x, y, w, h))
                    
            return obstacles
            
        except Exception as e:
            logger.error(f"Error detecting obstacles: {e}")
            return []


class SimpleSteeringPredictor:
    """Predicts steering commands based on the current frame"""
    
    def __init__(self):
        # Simple steering logic based on obstacle positions
        self.frame_center_x = None
        logger.info("Simple steering predictor initialized")
        
    def predict_steering(self, frame, obstacles):
        """
        Predict steering direction based on obstacles
        Returns: (direction, confidence)
        direction: "left", "right", "forward", or "stop"
        confidence: 0.0 to 1.0
        """
        if self.frame_center_x is None:
            self.frame_center_x = frame.shape[1] // 2
            
        if not obstacles:
            # No obstacles, go forward
            return "forward", 0.9
            
        # Calculate the "center of mass" of obstacles
        total_area = 0
        weighted_x = 0
        
        for x, y, w, h in obstacles:
            area = w * h
            center_x = x + w // 2
            weighted_x += center_x * area
            total_area += area
            
        if total_area > 0:
            obstacle_center_x = weighted_x / total_area
            
            # If obstacles are taking up too much space, stop
            if total_area > (frame.shape[0] * frame.shape[1]) * 0.3:
                return "stop", 0.9
                
            # Determine direction based on obstacle position
            if obstacle_center_x < self.frame_center_x:
                # Obstacles are more to the left, go right
                return "right", 0.7
            else:
                # Obstacles are more to the right, go left
                return "left", 0.7
        
        # Default to forward if no obstacles or calculation failed
        return "forward", 0.5


class AutonomousNavigator:
    """Handles autonomous navigation using camera input"""
    
    def __init__(self, motor_controller, camera_controller):
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.obstacle_detector = SimpleObstacleDetector()
        self.steering_predictor = SimpleSteeringPredictor()
        self.memory_buffer = MemoryBuffer()
        
        self.running = False
        self.navigation_thread = None
        self.data_collection_enabled = False
        self.last_state = None
        self.last_action = None
        
        # Navigation parameters
        self.speed = 0.4  # Default speed
        self.turn_speed = 0.5  # Speed when turning
        self.stop_distance = 0.2  # Normalized distance to stop
        
        logger.info("Autonomous navigator initialized")
        
    def start_navigation(self):
        """Start autonomous navigation"""
        if self.navigation_thread is None or not self.navigation_thread.is_alive():
            self.running = True
            self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
            self.navigation_thread.start()
            logger.info("Autonomous navigation started")
            return True
        return False
        
    def stop_navigation(self):
        """Stop autonomous navigation"""
        self.running = False
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)
            self.motor_controller.move("stop", 0)
            logger.info("Autonomous navigation stopped")
            return True
        return False
        
    def set_speed(self, speed):
        """Set the navigation speed"""
        self.speed = max(0.1, min(1.0, float(speed)))
        logger.info(f"Navigation speed set to {self.speed}")
        
    def toggle_data_collection(self, enabled=None):
        """Toggle data collection for training"""
        if enabled is not None:
            self.data_collection_enabled = enabled
        else:
            self.data_collection_enabled = not self.data_collection_enabled
            
        logger.info(f"Data collection {'enabled' if self.data_collection_enabled else 'disabled'}")
        return self.data_collection_enabled
        
    def save_memory_buffer(self):
        """Save the current memory buffer to disk"""
        self.memory_buffer.save()
        
    def _navigation_loop(self):
        """Main loop for autonomous navigation"""
        while self.running:
            try:
                # Get current frame
                with self.camera_controller.frame_lock:
                    if self.camera_controller.latest_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.camera_controller.latest_frame.copy()
                
                # Detect obstacles
                obstacles = self.obstacle_detector.detect_obstacles(frame)
                
                # Predict steering
                direction, confidence = self.steering_predictor.predict_steering(frame, obstacles)
                
                # Current state (simplified representation of the frame)
                current_state = self._preprocess_frame(frame)
                
                # Execute steering command
                if direction == "forward":
                    self.motor_controller.move("forward", self.speed)
                elif direction == "left":
                    self.motor_controller.move("left", self.turn_speed)
                elif direction == "right":
                    self.motor_controller.move("right", self.turn_speed)
                elif direction == "stop":
                    self.motor_controller.move("stop", 0)
                
                # Collect data if enabled
                if self.data_collection_enabled and self.last_state is not None:
                    # Simple reward: positive for forward movement, negative for stopping
                    reward = 1.0 if direction == "forward" else (0.5 if direction in ["left", "right"] else -0.5)
                    
                    # Add experience to memory buffer
                    self.memory_buffer.add_experience(
                        self.last_state,
                        self.last_action,
                        reward,
                        current_state,
                        direction == "stop"
                    )
                
                # Update last state and action
                self.last_state = current_state
                self.last_action = direction
                
                # Log navigation decisions
                logger.debug(f"Navigation: direction={direction}, confidence={confidence:.2f}, obstacles={len(obstacles)}")
                
                time.sleep(0.1)  # Control loop rate
                
            except Exception as e:
                logger.error(f"Error in navigation loop: {e}")
                time.sleep(0.5)
                
        # Ensure motors are stopped when exiting
        self.motor_controller.move("stop", 0)
    
    def _preprocess_frame(self, frame):
        """
        Preprocess frame for memory buffer storage
        Reduces size and converts to grayscale to save memory
        """
        # Resize to smaller dimensions
        small_frame = cv2.resize(frame, (80, 60))
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        return gray
    
    def get_navigation_info(self):
        """Get information about the current navigation state"""
        return {
            "active": self.running,
            "data_collection": self.data_collection_enabled,
            "memory_size": len(self.memory_buffer),
            "speed": self.speed
        }
        
    def cleanup(self):
        """Clean up resources"""
        self.stop_navigation()
        # Save memory buffer before exiting
        if len(self.memory_buffer) > 0:
            self.memory_buffer.save()
        logger.info("Autonomous navigator cleaned up")


def draw_navigation_overlay(frame, obstacles, direction, confidence):
    """Draw navigation information on the frame"""
    # Draw detected obstacles
    for x, y, w, h in obstacles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Draw direction indicator
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height - 50
    
    # Draw circle background
    cv2.circle(frame, (center_x, center_y), 30, (0, 0, 0), -1)
    
    # Draw direction arrow
    if direction == "forward":
        pts = np.array([[center_x, center_y - 20], [center_x - 10, center_y + 10], [center_x + 10, center_y + 10]])
        cv2.fillPoly(frame, [pts], (0, 255, 0))
    elif direction == "left":
        pts = np.array([[center_x - 20, center_y], [center_x + 10, center_y - 10], [center_x + 10, center_y + 10]])
        cv2.fillPoly(frame, [pts], (0, 255, 0))
    elif direction == "right":
        pts = np.array([[center_x + 20, center_y], [center_x - 10, center_y - 10], [center_x - 10, center_y + 10]])
        cv2.fillPoly(frame, [pts], (0, 255, 0))
    elif direction == "stop":
        cv2.rectangle(frame, (center_x - 10, center_y - 10), (center_x + 10, center_y + 10), (0, 0, 255), -1)
    
    # Draw confidence text
    cv2.putText(frame, f"Conf: {confidence:.2f}", (center_x - 40, center_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


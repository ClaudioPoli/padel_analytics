"""Utilities for padel court keypoints detection and editing."""
import cv2
import numpy as np
from typing import List, Tuple

def draw_keypoints(image: np.ndarray, keypoints: List[Tuple[float, float]], 
                   labels: bool = True, connections: bool = True) -> np.ndarray:
    """
    Draw keypoints on an image with optional labels and connections between points.
    
    Args:
        image: The input image (OpenCV format)
        keypoints: List of (x, y) keypoint coordinates
        labels: Whether to draw keypoint labels (default: True)
        connections: Whether to draw connections between keypoints (default: True)
        
    Returns:
        Image with drawn keypoints and connections
    """
    img_copy = image.copy()
    
    # Define connections for padel court
    court_connections = [
        (0, 1),  # Bottom horizontal line (k1-k2)
        (0, 2),  # Left vertical line bottom section (k1-k3)
        (1, 4),  # Right vertical line bottom section (k2-k5)
        (2, 3),  # Middle horizontal line bottom section (k3-k4)
        (3, 4),  # Right horizontal line bottom section (k4-k5)
        (2, 5),  # Left vertical line middle section (k3-k6)
        (4, 6),  # Right vertical line middle section (k5-k7)
        (5, 6),  # Middle horizontal line middle section (k6-k7)
        (5, 7),  # Left vertical line top section (k6-k8)
        (6, 9),  # Right vertical line top section (k7-k10)
        (7, 8),  # Middle horizontal line top section (k8-k9)
        (8, 9),  # Right horizontal line top section (k9-k10)
        (7, 10),  # Left vertical line top (k8-k11)
        (9, 11),  # Right vertical line top (k10-k12)
        (10, 11),  # Top horizontal line (k11-k12)
    ]

    # Draw connections
    if connections and len(keypoints) >= 12:
        for start_idx, end_idx in court_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_x, start_y = int(keypoints[start_idx][0]), int(keypoints[start_idx][1])
                end_x, end_y = int(keypoints[end_idx][0]), int(keypoints[end_idx][1])
                cv2.line(img_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        
        # Draw circle for keypoint
        cv2.circle(img_copy, (x, y), radius=6, color=(255, 0, 0), thickness=-1)
        
        if labels:
            # Draw keypoint label
            cv2.putText(
                img_copy, 
                f"k{i+1}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    
    return img_copy


def manual_keypoints_selection(image: np.ndarray, initial_keypoints: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """
    Interactive tool to manually select or edit keypoints.
    
    Args:
        image: The input image
        initial_keypoints: Optional list of initial keypoints to edit
        
    Returns:
        List of (x, y) keypoint coordinates selected/edited by user
    """
    selected_keypoints = [] if initial_keypoints is None else list(initial_keypoints)
    editing_mode = initial_keypoints is not None
    edit_radius = 20  # Radius in pixels for selecting a point to edit
    current_edit_index = None
    
    # Create window
    cv2.namedWindow('Padel Court Keypoints Selection')
    
    # Instructions text
    instructions = (
        "Left-click: Add/move keypoint | "
        "Right-click: Delete keypoint | "
        "Space: Finish selection"
    )
    
    def update_display():
        """Update the display with current keypoints"""
        display_img = draw_keypoints(image, selected_keypoints)
        
        # Add instruction text
        cv2.putText(
            display_img,
            instructions,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Show number of keypoints
        cv2.putText(
            display_img,
            f"Keypoints: {len(selected_keypoints)}/12",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        cv2.imshow('Padel Court Keypoints Selection', display_img)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_edit_index
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find if we're clicking near an existing point (for editing)
            if editing_mode:
                for i, (kp_x, kp_y) in enumerate(selected_keypoints):
                    if ((x - kp_x) ** 2 + (y - kp_y) ** 2) <= edit_radius ** 2:
                        current_edit_index = i
                        break
                else:
                    # No nearby point found, add new one
                    if len(selected_keypoints) < 12:
                        selected_keypoints.append((float(x), float(y)))
            else:
                # In add mode, just add points
                if len(selected_keypoints) < 12:
                    selected_keypoints.append((float(x), float(y)))
                    
            update_display()
            
        elif event == cv2.EVENT_MOUSEMOVE and current_edit_index is not None:
            # Move the selected point
            selected_keypoints[current_edit_index] = (float(x), float(y))
            update_display()
            
        elif event == cv2.EVENT_LBUTTONUP:
            current_edit_index = None
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the nearest keypoint
            if selected_keypoints:
                distances = [(i, (x - kp_x) ** 2 + (y - kp_y) ** 2) 
                             for i, (kp_x, kp_y) in enumerate(selected_keypoints)]
                idx, _ = min(distances, key=lambda d: d[1])
                del selected_keypoints[idx]
                update_display()
    
    # Set mouse callback
    cv2.setMouseCallback('Padel Court Keypoints Selection', mouse_callback)
    
    # Initial display
    update_display()
    
    # Wait for user to finish
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space key
            break
    
    cv2.destroyAllWindows()
    return selected_keypoints

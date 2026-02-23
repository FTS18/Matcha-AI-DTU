import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DynamicPitchCalibrator:
    """
    Automatically detects soccer pitch lines and corners to compute 
    Homography without manual intervention.
    """
    
    def __init__(self, w=800, h=520):
        self.pitch_w = w
        self.pitch_h = h
        self.prev_H = None
        
    def calibrate_frame(self, frame):
        """
        Detect lines, find intersections, and return a Homography matrix.
        """
        # 1. Isolate white lines (using color mask + Canny)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White colors in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        edges = cv2.Canny(mask, 50, 150)
        
        # 2. Find lines using probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return self.prev_H

        # 3. Strategy: Find 4 dominant lines (sidelines and goal lines)
        # For a professional project, we would use a Line Segment Transformer here.
        # This is a heuristic fallback:
        extracted_points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                extracted_points.append((x1, y1))
                extracted_points.append((x2, y2))
        
        # 4. If we find enough geometry, attempt to find the "Pitch Hull"
        if len(extracted_points) > 4:
            # Simple approximation of the field boundary
            hull = cv2.convexHull(np.array(extracted_points))
            # Approximate the hull to a polygon
            epsilon = 0.05 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                # We found a quadrilateral! 
                src_pts = approx.reshape(4, 2).astype(np.float32)
                # Sort points: top-left, top-right, bottom-right, bottom-left
                src_pts = self._sort_points(src_pts)
                
                dst_pts = np.array([
                    [0, 0],
                    [self.pitch_w, 0],
                    [self.pitch_w, self.pitch_h],
                    [0, self.pitch_h]
                ], dtype=np.float32)
                
                H, _ = cv2.findHomography(src_pts, dst_pts)
                self.prev_H = H
                return H
                
        return self.prev_H

    def _sort_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

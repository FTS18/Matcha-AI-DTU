import cv2
import numpy as np
import json
import argparse
from pathlib import Path


def calibrate_pitch(image_path, output_json):
    """
    Open an image and click 4 points in the following order:
    1. Top-Left (e.g., Corner Flag or Penalty Box corner)
    2. Top-Right
    3. Bottom-Right
    4. Bottom-Left

    Then defines the real-world metrics (105m x 68m).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    clone = img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"{len(points)}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Calibrator", img)

    if len(points) == 4:
        print(" 4 points collected.")
        print("Press 'S' to save or 'R' to reset.")

    cv2.namedWindow("Calibrator")
    cv2.setMouseCallback("Calibrator", click_event)

    print("--- PITCH CALIBRATION TOOL ---")
    print("Click 4 points on the pitch boundary in this order:")
    print("1. Top-Left 2. Top-Right 3. Bottom-Right 4. Bottom-Left")
    print("Then press 'S' to save the mapping.")

    while True:
        cv2.imshow("Calibrator", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            img = clone.copy()
            points = []
            print("Resetting points...")

        elif key == ord("s") and len(points) == 4:
            # Source points (from image)
            src_pts = np.array(points, dtype=np.float32)

            # Destination points (canonical pitch 105x68 converted to pixels for display,
            # but we save normalized coords usually)
            dst_w, dst_h = 800, 520
            dst_pts = np.array(
                [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32
            )

            # Compute homography
            H, _ = cv2.findHomography(src_pts, dst_pts)

            # Save to JSON
            calibration_data = {
                "source_points": src_pts.tolist(),
                "dest_points": dst_pts.tolist(),
                "homography_matrix": H.tolist(),
                "pitch_dimensions": {"width_m": 105, "height_m": 68},
            }

            with open(output_json, "w") as f:
                json.dump(calibration_data, f, indent=4)

            print(f" Calibration saved to {output_json}")
            break

        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Pitch Homography")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to a frame image"
    )
    parser.add_argument(
        "--output", type=str, default="homography.json", help="Output JSON filename"
    )

    args = parser.parse_args()
    calibrate_pitch(args.image, args.output)

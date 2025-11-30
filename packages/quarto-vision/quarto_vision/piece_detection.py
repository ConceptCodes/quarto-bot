#!/usr/bin/env python3
"""
Unified Piece Detection Script for Quarto Bot
Handles detection from images, live camera feed, or robot camera integration
"""

import sys
import os
import argparse
import cv2
import json
import time
from pathlib import Path

from ultralytics import YOLO
from quarto_driver.camera_feed import CameraFeed


class QuartoPieceDetector:
    """Unified piece detection for images, camera feed, and robot integration."""

    def __init__(self, model_path=None, confidence=0.5):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model (uses default if None)
            confidence: Default confidence threshold
        """
        if model_path is None:
            model_path = "runs/detect/quarto_model_optimized/weights/best.pt"

        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the YOLO model."""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading YOLO model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                print("✓ Model loaded successfully!")
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def detect_from_image(
        self, image_path, confidence=None, save_results=True, show_results=False
    ):
        """
        Detect pieces from a single image.

        Args:
            image_path: Path to image file
            confidence: Detection confidence (uses default if None)
            save_results: Whether to save annotated image
            show_results: Whether to display results

        Returns:
            dict: Detection results with structured data
        """
        if confidence is None:
            confidence = self.confidence

        print(f"Detecting pieces in: {image_path}")
        print(f"Confidence threshold: {confidence}")

        # Run YOLO detection
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            show=show_results,
            save=False,  # Don't use YOLO's default save
            verbose=False,
        )

        # Process results
        detection_data = self._process_results(results, source=image_path)

        # Save annotated image to assets folder with model name suffix
        if save_results:
            self._save_annotated_image(results, image_path)

        return detection_data

    def _save_annotated_image(self, results, image_path):
        """Save annotated image to assets folder with model name suffix."""
        try:
            # Load original image
            import cv2

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return

            # Draw annotations
            for result in results:
                boxes = result.boxes
                names = result.names
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        label = f"{names[class_id]} {conf_val:.2f}"

                        # Use different colors for different classes
                        colors = [
                            (255, 0, 0),  # Blue
                            (0, 255, 0),  # Green
                            (0, 0, 255),  # Red
                            (255, 255, 0),  # Cyan
                            (255, 0, 255),  # Magenta
                            (0, 255, 255),  # Yellow
                            (128, 0, 128),  # Purple
                            (255, 165, 0),  # Orange
                        ]
                        box_color = colors[class_id % len(colors)]
                        text_color = (255, 255, 255)  # White text

                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 4)

                        # Get text size for background rectangle
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5  # Even larger font
                        thickness = 3
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )

                        # Draw colored background rectangle for text (same color as box)
                        cv2.rectangle(
                            image,
                            (x1, y1 - text_height - 15),
                            (x1 + text_width + 10, y1),
                            box_color,
                            -1,
                        )

                        # Draw text
                        cv2.putText(
                            image,
                            label,
                            (x1 + 5, y1 - 8),
                            font,
                            font_scale,
                            text_color,
                            thickness,
                        )

            # Create output filename with model name suffix
            input_path = Path(image_path)
            model_name = Path(
                self.model_path
            ).parent.parent.name  # Extract model name from path
            output_filename = (
                f"{input_path.stem}_{model_name}_detected{input_path.suffix}"
            )
            output_path = Path("assets") / output_filename

            # Ensure assets directory exists
            output_path.parent.mkdir(exist_ok=True)

            # Save annotated image
            cv2.imwrite(str(output_path), image)
            print(f"✓ Annotated image saved to: {output_path}")

        except Exception as e:
            print(f"Failed to save annotated image: {e}")

    def detect_from_camera_single(
        self, camera_index=0, confidence=None, save_snapshot=True
    ):
        """
        Take a single snapshot from camera and detect pieces.
        Perfect for robot integration - capture and analyze.

        Args:
            camera_index: Camera device index
            confidence: Detection confidence (uses default if None)
            save_snapshot: Whether to save the snapshot

        Returns:
            dict: Detection results with structured data
        """
        if confidence is None:
            confidence = self.confidence

        print(f"Capturing from camera {camera_index} for detection...")

        camera = CameraFeed(camera_index=camera_index)

        try:
            if camera.start_feed():
                print("✓ Camera started")

                # Wait for camera to stabilize
                time.sleep(2)

                # Capture frame
                frame = camera.get_frame()
                if frame is None:
                    raise Exception("Failed to capture frame")

                print(f"✓ Frame captured: {frame.shape}")

                # Save snapshot if requested
                snapshot_path = None
                if save_snapshot:
                    timestamp = int(time.time())
                    snapshot_path = f"robot_capture_{timestamp}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    print(f"✓ Snapshot saved: {snapshot_path}")

                # Run detection on frame
                results = self.model.predict(
                    source=frame, conf=confidence, show=False, save=False, verbose=False
                )

                # Process results
                detection_data = self._process_results(results, source="camera")
                detection_data["snapshot_path"] = snapshot_path

                return detection_data

            else:
                raise Exception("Failed to start camera")

        finally:
            camera.stop_feed()

    def detect_live_feed(self, camera_index=0, confidence=None):
        """
        Run live detection with bounding boxes overlay.

        Args:
            camera_index: Camera device index
            confidence: Detection confidence (uses default if None)
        """
        if confidence is None:
            confidence = self.confidence

        print(f"Starting live detection (confidence: {confidence})")

        camera = CameraFeed(camera_index=camera_index)

        try:
            if camera.start_feed():
                print("✓ Camera started for live detection")
                time.sleep(2)
                print("Press 'q' to quit live detection.")
                while True:
                    frame = camera.get_frame()
                    if frame is None:
                        print("Failed to get frame from camera.")
                        break
                    # Run YOLO detection
                    results = self.model.predict(
                        source=frame,
                        conf=confidence,
                        show=False,
                        save=False,
                        verbose=False,
                    )
                    # Draw bounding boxes
                    for result in results:
                        boxes = result.boxes
                        names = result.names
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                class_id = int(box.cls[0])
                                conf_val = float(box.conf[0])
                                label = f"{names[class_id]} {conf_val:.2f}"

                                # Use different colors for different classes
                                colors = [
                                    (255, 0, 0),  # Blue
                                    (0, 255, 0),  # Green
                                    (0, 0, 255),  # Red
                                    (255, 255, 0),  # Cyan
                                    (255, 0, 255),  # Magenta
                                    (0, 255, 255),  # Yellow
                                    (128, 0, 128),  # Purple
                                    (255, 165, 0),  # Orange
                                ]
                                box_color = colors[class_id % len(colors)]
                                text_color = (255, 255, 255)  # White text

                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                                # Get text size for background rectangle
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1.5  # Even larger font
                                thickness = 3
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label, font, font_scale, thickness
                                )

                                # Draw colored background rectangle for text (same color as box)
                                cv2.rectangle(
                                    frame,
                                    (x1, y1 - text_height - 15),
                                    (x1 + text_width + 10, y1),
                                    box_color,
                                    -1,
                                )

                                # Draw text
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1 + 5, y1 - 8),
                                    font,
                                    font_scale,
                                    text_color,
                                    thickness,
                                )
                    cv2.imshow(f"Live Detection (index {camera_index})", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cv2.destroyAllWindows()
            else:
                print("✗ Failed to start camera")
        finally:
            camera.stop_feed()

    def _process_results(self, results, source="unknown"):
        """
        Process YOLO results into structured data.

        Returns:
            dict: Structured detection data
        """
        detection_data = {
            "source": source,
            "timestamp": int(time.time()),
            "total_detections": 0,
            "pieces": [],
            "piece_types": {},
            "confidence_threshold": self.confidence,
        }

        if not results or len(results) == 0:
            return detection_data

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                detection_data["total_detections"] = len(boxes)

                for box in boxes:
                    piece_info = {
                        "class_id": int(box.cls[0]),
                        "class_name": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x1": float(box.xyxy[0][0]),
                            "y1": float(box.xyxy[0][1]),
                            "x2": float(box.xyxy[0][2]),
                            "y2": float(box.xyxy[0][3]),
                        },
                        "center": {
                            "x": float((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                            "y": float((box.xyxy[0][1] + box.xyxy[0][3]) / 2),
                        },
                        "area": float(
                            (box.xyxy[0][2] - box.xyxy[0][0])
                            * (box.xyxy[0][3] - box.xyxy[0][1])
                        ),
                    }

                    detection_data["pieces"].append(piece_info)

                    # Count piece types
                    piece_type = piece_info["class_name"]
                    if piece_type not in detection_data["piece_types"]:
                        detection_data["piece_types"][piece_type] = 0
                    detection_data["piece_types"][piece_type] += 1

        return detection_data

    def save_detection_data(self, detection_data, output_path=None):
        """
        Save detection data to JSON file.

        Args:
            detection_data: Detection results from any detect method
            output_path: Output file path (auto-generated if None)

        Returns:
            str: Path to saved file
        """
        if output_path is None:
            timestamp = detection_data.get("timestamp", int(time.time()))
            output_path = f"detection_results_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(detection_data, f, indent=2)

        print(f"✓ Detection data saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Quarto Piece Detection - Images, Camera, or Live Feed"
    )

    # Mode selection
    parser.add_argument(
        "mode",
        choices=["image", "camera", "live"],
        help="Detection mode: image (single image), camera (single capture), live (interactive feed)",
    )

    # Source specification
    parser.add_argument(
        "source",
        nargs="?",
        help="Image path (for image mode) or camera index (for camera/live modes, default: 0)",
    )

    # Common options
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model (uses default if not specified)",
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save detection results to JSON file"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save images/snapshots"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show detection results (for image mode)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "image" and not args.source:
        parser.error("Image path required for image mode")

    # Set default camera index
    if args.mode in ["camera", "live"] and not args.source:
        args.source = "0"

    # Initialize detector
    try:
        detector = QuartoPieceDetector(
            model_path=args.model, confidence=args.confidence
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return 1

    # Run detection based on mode
    try:
        if args.mode == "image":
            print(f"\n=== Image Detection Mode ===")
            detection_data = detector.detect_from_image(
                image_path=args.source,
                confidence=args.confidence,
                save_results=not args.no_save,
                show_results=args.show,
            )

        elif args.mode == "camera":
            print(f"\n=== Camera Capture Mode ===")
            camera_index = int(args.source) if args.source.isdigit() else 0
            detection_data = detector.detect_from_camera_single(
                camera_index=camera_index,
                confidence=args.confidence,
                save_snapshot=not args.no_save,
            )

        elif args.mode == "live":
            print(f"\n=== Live Detection Mode ===")
            camera_index = int(args.source) if args.source.isdigit() else 0
            detector.detect_live_feed(
                camera_index=camera_index, confidence=args.confidence
            )
            return 0  # Live mode doesn't return detection data

        # Print results summary
        print(f"\n=== Detection Results ===")
        print(f"Total detections: {detection_data['total_detections']}")
        print(f"Piece types found:")
        for piece_type, count in detection_data["piece_types"].items():
            print(f"  {piece_type}: {count}")

        if detection_data["pieces"]:
            print(f"\nDetailed results:")
            for i, piece in enumerate(detection_data["pieces"], 1):
                print(
                    f"  Piece {i}: {piece['class_name']} "
                    f"(confidence: {piece['confidence']:.2f}, "
                    f"center: {piece['center']['x']:.0f}, {piece['center']['y']:.0f})"
                )

        # Save JSON if requested
        if args.save_json:
            detector.save_detection_data(detection_data)

        return 0

    except Exception as e:
        print(f"Detection failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

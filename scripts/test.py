#!/usr/bin/env python3
"""
Pool Ball Detection Script
Handles detection from images, live camera feed, or video files
"""

import os
import argparse
import cv2
import json
import time

from ultralytics import YOLO


class PoolBallDetector:
    """Unified ball detection for images, camera feed, and video analysis."""

    def __init__(self, model_path=None, confidence=0.5):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model (uses default if None)
            confidence: Default confidence threshold
        """
        if model_path is None:
            model_path = "runs/detect/pool-ball-detection/weights/best.pt"

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
        Detect pool balls from a single image.

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

        print(f"Detecting pool balls in: {image_path}")
        print(f"Confidence threshold: {confidence}")

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            show=show_results,
            save=save_results,
            verbose=False,
        )

        detection_data = self._process_results(results, source=image_path)

        if save_results:
            print(f"✓ Annotated image saved to: runs/detect/predict/")

        return detection_data

    def detect_from_video(
        self, video_path, confidence=None, save_video=True, show_live=False
    ):
        """
        Detect pool balls from video file.

        Args:
            video_path: Path to video file
            confidence: Detection confidence (uses default if None)
            save_video: Whether to save annotated video
            show_live: Whether to display video while processing

        Returns:
            dict: Detection results with frame-by-frame data
        """
        if confidence is None:
            confidence = self.confidence

        print(f"Processing video: {video_path}")
        print(f"Confidence threshold: {confidence}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames"
        )

        out = None
        if save_video:
            timestamp = int(time.time())
            output_path = f"annotated_video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        video_data = {
            "source": video_path,
            "timestamp": int(time.time()),
            "video_properties": {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
            },
            "frames": [],
            "summary": {"total_balls_detected": 0, "frames_with_balls": 0},
        }

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=confidence,
                show=False,
                save=False,
                verbose=False,
            )

            frame_data = self._process_results(results, source=f"frame_{frame_count}")
            frame_data["frame_number"] = frame_count
            frame_data["timestamp"] = frame_count / fps

            video_data["frames"].append(frame_data)

            if frame_data["total_detections"] > 0:
                video_data["summary"]["frames_with_balls"] += 1
                video_data["summary"]["total_balls_detected"] += frame_data[
                    "total_detections"
                ]

            # Draw annotations on frame
            annotated_frame = frame.copy()
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        label = f"{result.names[class_id]} {conf_val:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            if out is not None:
                out.write(annotated_frame)

            if show_live:
                cv2.imshow("Pool Ball Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        cap.release()
        if out is not None:
            out.release()
            print(f"✓ Annotated video saved to: {output_path}")
        if show_live:
            cv2.destroyAllWindows()

        print(f"Video processing complete!")
        print(
            f"Frames with balls: {video_data['summary']['frames_with_balls']}/{total_frames}"
        )
        print(f"Total ball detections: {video_data['summary']['total_balls_detected']}")

        return video_data

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

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")

        print("✓ Camera started for live detection")
        print("Press 'q' to quit live detection.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
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
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

                cv2.imshow(f"Live Pool Ball Detection (Camera {camera_index})", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

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
            "balls": [],
            "ball_types": {},
            "confidence_threshold": self.confidence,
        }

        if not results or len(results) == 0:
            return detection_data

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                detection_data["total_detections"] = len(boxes)

                for box in boxes:
                    ball_info = {
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

                    detection_data["balls"].append(ball_info)

                    # Count ball types
                    ball_type = ball_info["class_name"]
                    if ball_type not in detection_data["ball_types"]:
                        detection_data["ball_types"][ball_type] = 0
                    detection_data["ball_types"][ball_type] += 1

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
            output_path = f"pool_detection_results_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(detection_data, f, indent=2)

        print(f"✓ Detection data saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pool Ball Detection - Images, Video, Camera, or Live Feed"
    )

    parser.add_argument(
        "mode",
        choices=["image", "video", "camera", "live"],
        help="Detection mode: image (single image), video (video file), camera (single capture), live (interactive feed)",
    )

    parser.add_argument(
        "source",
        nargs="?",
        help="Image/video path (for image/video mode) or camera index (for camera/live modes, default: 0)",
    )

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
        "--no-save", action="store_true", help="Don't save images/videos/snapshots"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show detection results during processing"
    )

    args = parser.parse_args()

    if args.mode in ["image", "video"] and not args.source:
        parser.error(f"{args.mode.capitalize()} path required for {args.mode} mode")

    if args.mode in ["camera", "live"] and not args.source:
        args.source = "0"

    try:
        detector = PoolBallDetector(model_path=args.model, confidence=args.confidence)
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

        elif args.mode == "video":
            print(f"\n=== Video Detection Mode ===")
            detection_data = detector.detect_from_video(
                video_path=args.source,
                confidence=args.confidence,
                save_video=not args.no_save,
                show_live=args.show,
            )

        elif args.mode == "camera":
            print(f"\n=== Camera Capture Mode ===")
            camera_index = int(args.source) if args.source.isdigit() else 0
            # For camera mode, we'll do a live feed since single capture doesn't make as much sense for pool
            detector.detect_live_feed(
                camera_index=camera_index, confidence=args.confidence
            )
            return 0

        elif args.mode == "live":
            print(f"\n=== Live Detection Mode ===")
            camera_index = int(args.source) if args.source.isdigit() else 0
            detector.detect_live_feed(
                camera_index=camera_index, confidence=args.confidence
            )
            return 0  # Live mode doesn't return detection data

        # Print results summary (for image and video modes)
        if args.mode in ["image", "video"]:
            if args.mode == "image":
                print(f"\n=== Detection Results ===")
                print(f"Total detections: {detection_data['total_detections']}")
                print(f"Ball types found:")
                for ball_type, count in detection_data["ball_types"].items():
                    print(f"  {ball_type}: {count}")

                if detection_data["balls"]:
                    print(f"\nDetailed results:")
                    for i, ball in enumerate(detection_data["balls"], 1):
                        print(
                            f"  Ball {i}: {ball['class_name']} "
                            f"(confidence: {ball['confidence']:.2f}, "
                            f"center: {ball['center']['x']:.0f}, {ball['center']['y']:.0f})"
                        )
            else:  # video mode
                print(f"\n=== Video Detection Summary ===")
                print(f"Total frames processed: {len(detection_data['frames'])}")
                print(
                    f"Frames with ball detections: {detection_data['summary']['frames_with_balls']}"
                )
                print(
                    f"Total ball detections: {detection_data['summary']['total_balls_detected']}"
                )

            if args.save_json:
                detector.save_detection_data(detection_data)

        return 0

    except Exception as e:
        print(f"Detection failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

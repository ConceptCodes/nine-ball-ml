#!/usr/bin/env python3
"""
Frame Extraction Script for Computer Vision
Extracts frames from video with options for selective extraction
"""

import cv2
import os
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional


def get_video_info(video_path: str) -> dict:
    """Get basic information about the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }


def time_to_frame(time_seconds: float, fps: float) -> int:
    """Convert time in seconds to frame number."""
    return int(time_seconds * fps)


def frame_to_time(frame_number: int, fps: float) -> float:
    """Convert frame number to time in seconds."""
    return frame_number / fps if fps > 0 else 0


def preview_frames(video_path: str, start_frame: int = 0, end_frame: Optional[int] = None, 
                  step: int = 30) -> None:
    """Preview frames from the video to help identify useful segments."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Previewing frames from {start_frame} to {end_frame} (step: {step})")
    print("Press 'q' to quit preview, 's' to skip to next frame, any other key to continue")
    
    for frame_num in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Add frame info overlay
        time_sec = frame_to_time(frame_num, fps)
        info_text = f"Frame: {frame_num} | Time: {time_sec:.2f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for display if too large
        height, width = frame.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow('Frame Preview', frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            continue
    
    cap.release()
    cv2.destroyAllWindows()


def extract_frames(video_path: str, output_dir: str, 
                  time_ranges: List[Tuple[float, float]] = None,
                  frame_ranges: List[Tuple[int, int]] = None,
                  step: int = 1, 
                  max_frames: Optional[int] = None,
                  prefix: str = "frame") -> List[str]:
    """
    Extract frames from video with various filtering options.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        time_ranges: List of (start_time, end_time) tuples in seconds
        frame_ranges: List of (start_frame, end_frame) tuples
        step: Extract every Nth frame (default: 1 for every frame)
        max_frames: Maximum number of frames to extract
        prefix: Prefix for saved frame filenames
    
    Returns:
        List of saved frame file paths
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert time ranges to frame ranges if provided
    if time_ranges:
        if frame_ranges is None:
            frame_ranges = []
        for start_time, end_time in time_ranges:
            start_frame = time_to_frame(start_time, fps)
            end_frame = time_to_frame(end_time, fps)
            frame_ranges.append((start_frame, end_frame))
    
    # If no ranges specified, extract from entire video
    if not frame_ranges:
        frame_ranges = [(0, total_frames)]
    
    saved_files = []
    frames_extracted = 0
    
    print(f"Extracting frames from {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame ranges: {frame_ranges}")
    print(f"Step: {step}")
    
    for start_frame, end_frame in frame_ranges:
        print(f"\nProcessing range: frames {start_frame} to {end_frame}")
        
        for frame_num in range(start_frame, min(end_frame, total_frames), step):
            if max_frames and frames_extracted >= max_frames:
                print(f"Reached maximum frames limit: {max_frames}")
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_num}")
                continue
            
            # Generate filename
            time_sec = frame_to_time(frame_num, fps)
            filename = f"{prefix}_{frame_num:06d}_t{time_sec:.2f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            success = cv2.imwrite(filepath, frame)
            if success:
                saved_files.append(filepath)
                frames_extracted += 1
                
                if frames_extracted % 100 == 0:
                    print(f"Extracted {frames_extracted} frames...")
            else:
                print(f"Failed to save frame {frame_num}")
        
        if max_frames and frames_extracted >= max_frames:
            break
    
    cap.release()
    print(f"\nExtraction complete! Saved {frames_extracted} frames to {output_dir}")
    return saved_files


def interactive_extraction(video_path: str, output_dir: str) -> None:
    """Interactive mode for frame extraction with preview."""
    info = get_video_info(video_path)
    print(f"\nVideo Information:")
    print(f"Duration: {info['duration']:.2f} seconds")
    print(f"FPS: {info['fps']:.2f}")
    print(f"Total frames: {info['frame_count']}")
    print(f"Resolution: {info['width']}x{info['height']}")
    
    while True:
        print("\n" + "="*50)
        print("Frame Extraction Options:")
        print("1. Preview frames")
        print("2. Extract frames by time range")
        print("3. Extract frames by frame range")
        print("4. Extract every Nth frame from entire video")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            start = input(f"Start frame (0-{info['frame_count']-1}, default 0): ").strip()
            start_frame = int(start) if start else 0
            
            end = input(f"End frame (default {info['frame_count']}): ").strip()
            end_frame = int(end) if end else info['frame_count']
            
            step = input("Preview step (default 30): ").strip()
            preview_step = int(step) if step else 30
            
            preview_frames(video_path, start_frame, end_frame, preview_step)
        
        elif choice == '2':
            time_ranges = []
            while True:
                start_time = input(f"Start time in seconds (0-{info['duration']:.2f}): ")
                if not start_time:
                    break
                end_time = input(f"End time in seconds: ")
                if not end_time:
                    break
                
                time_ranges.append((float(start_time), float(end_time)))
                
                more = input("Add another time range? (y/n): ").strip().lower()
                if more != 'y':
                    break
            
            if time_ranges:
                step = input("Extract every Nth frame (default 1): ").strip()
                frame_step = int(step) if step else 1
                
                extract_frames(video_path, output_dir, time_ranges=time_ranges, step=frame_step)
        
        elif choice == '3':
            frame_ranges = []
            while True:
                start_frame = input(f"Start frame (0-{info['frame_count']-1}): ")
                if not start_frame:
                    break
                end_frame = input(f"End frame: ")
                if not end_frame:
                    break
                
                frame_ranges.append((int(start_frame), int(end_frame)))
                
                more = input("Add another frame range? (y/n): ").strip().lower()
                if more != 'y':
                    break
            
            if frame_ranges:
                step = input("Extract every Nth frame (default 1): ").strip()
                frame_step = int(step) if step else 1
                
                extract_frames(video_path, output_dir, frame_ranges=frame_ranges, step=frame_step)
        
        elif choice == '4':
            step = input("Extract every Nth frame (default 30): ").strip()
            frame_step = int(step) if step else 30
            
            max_frames = input("Maximum frames to extract (optional): ").strip()
            max_frames = int(max_frames) if max_frames else None
            
            extract_frames(video_path, output_dir, step=frame_step, max_frames=max_frames)
        
        elif choice == '5':
            break
        
        else:
            print("Invalid option. Please select 1-5.")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from video for computer vision')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', '-o', 
                       default='../data/raw',
                       help='Output directory for extracted frames (default: ../data/raw)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview frames before extraction')
    parser.add_argument('--step', '-s', type=int, default=1,
                       help='Extract every Nth frame (default: 1)')
    parser.add_argument('--start-time', type=float,
                       help='Start time in seconds')
    parser.add_argument('--end-time', type=float,
                       help='End time in seconds')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to extract')
    parser.add_argument('--prefix', default='frame',
                       help='Prefix for saved frame filenames (default: frame)')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Convert relative output directory to absolute path
    if not os.path.isabs(args.output_dir):
        script_dir = Path(__file__).parent
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    output_dir = str(output_dir)
    
    try:
        if args.interactive:
            interactive_extraction(args.video_path, output_dir)
        elif args.preview:
            preview_frames(args.video_path)
        else:
            time_ranges = None
            if args.start_time is not None and args.end_time is not None:
                time_ranges = [(args.start_time, args.end_time)]
            
            extract_frames(
                args.video_path,
                output_dir,
                time_ranges=time_ranges,
                step=args.step,
                max_frames=args.max_frames,
                prefix=args.prefix
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
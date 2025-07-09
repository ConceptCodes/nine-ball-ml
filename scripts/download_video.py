#!/usr/bin/env python3
"""
YouTube Video Download Script for Computer Vision
Downloads videos from YouTube with options for quality and format selection
"""

import os
import sys
import argparse
from pathlib import Path
import re

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not found. Please install it with: pip install yt-dlp")
    sys.exit(1)


class YouTubeDownloader:
    """YouTube video downloader with customizable options."""

    def __init__(self, output_dir="data/raw"):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename):
        """Remove invalid characters from filename."""
        filename = re.sub(r'[<>:"/\\|?*]', "", filename)
        filename = filename.replace(" ", "_")
        if len(filename) > 100:
            filename = filename[:100]
        return filename

    def download_video(
        self,
        url,
        quality="best",
        audio_only=False,
        custom_filename=None,
        start_time=None,
        end_time=None,
    ):
        """
        Download a YouTube video.

        Args:
            url: YouTube video URL
            quality: Video quality ('best', 'worst', '720p', '480p', etc.)
            audio_only: Download audio only
            custom_filename: Custom filename (without extension)
            start_time: Start time in seconds for partial download
            end_time: End time in seconds for partial download

        Returns:
            str: Path to downloaded file
        """
        print(f"Downloading video from: {url}")

        if custom_filename:
            filename = self.sanitize_filename(custom_filename)
            output_template = str(self.output_dir / f"{filename}.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(title)s.%(ext)s")

        ydl_opts = {
            "outtmpl": output_template,
            "format": self._get_format_selector(quality, audio_only),
        }

        if start_time is not None or end_time is not None:
            postprocessor = {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
            if start_time is not None:
                postprocessor["start_time"] = start_time
            if end_time is not None:
                postprocessor["end_time"] = end_time

            ydl_opts["postprocessors"] = [postprocessor]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0)

                print(f"Video: {title}")
                print(f"Duration: {duration // 60}:{duration % 60:02d}")

                ydl.download([url])

                if custom_filename:
                    downloaded_file = self._find_downloaded_file(filename)
                else:
                    sanitized_title = self.sanitize_filename(title)
                    downloaded_file = self._find_downloaded_file(sanitized_title)

                if downloaded_file:
                    print(f"✓ Video downloaded successfully: {downloaded_file}")
                    return downloaded_file
                else:
                    print("✗ Could not locate downloaded file")
                    return None

        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None

    def _get_format_selector(self, quality, audio_only):
        """Get format selector string for yt-dlp."""
        if audio_only:
            return "bestaudio/best"

        if quality == "best":
            return "best[ext=mp4]/best"
        elif quality == "worst":
            return "worst[ext=mp4]/worst"
        elif quality.endswith("p"):
            # Specific resolution like 720p, 480p
            height = quality[:-1]
            return f"best[height<={height}][ext=mp4]/best[height<={height}]/best[ext=mp4]/best"
        else:
            return "best[ext=mp4]/best"

    def _find_downloaded_file(self, base_filename):
        """Find the downloaded file with various possible extensions."""
        possible_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]

        for ext in possible_extensions:
            file_path = self.output_dir / f"{base_filename}{ext}"
            if file_path.exists():
                return str(file_path)

        for file_path in self.output_dir.glob(f"{base_filename}*"):
            if file_path.is_file():
                return str(file_path)

        return None

    def get_video_info(self, url):
        """Get information about a YouTube video without downloading."""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)

                video_info = {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", "Unknown"),
                    "description": (
                        info.get("description", "")[:200] + "..."
                        if info.get("description")
                        else ""
                    ),
                    "available_formats": [],
                }

                for fmt in info.get("formats", []):
                    if fmt.get("height"):
                        format_info = {
                            "format_id": fmt.get("format_id"),
                            "quality": f"{fmt.get('height')}p",
                            "ext": fmt.get("ext"),
                            "filesize": fmt.get("filesize"),
                        }
                        video_info["available_formats"].append(format_info)

                return video_info

        except Exception as e:
            print(f"✗ Failed to get video info: {e}")
            return None

    def download_playlist(self, url, quality="best", max_videos=None):
        """Download videos from a YouTube playlist."""
        print(f"Downloading playlist from: {url}")

        ydl_opts = {
            "outtmpl": str(self.output_dir / "%(playlist_index)02d_%(title)s.%(ext)s"),
            "format": self._get_format_selector(quality, False),
        }

        if max_videos:
            ydl_opts["playlistend"] = max_videos

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("✓ Playlist downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Playlist download failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for computer vision"
    )

    parser.add_argument("url", help="YouTube video or playlist URL")

    parser.add_argument(
        "--quality",
        "-q",
        choices=["best", "worst", "1080p", "720p", "480p", "360p"],
        default="best",
        help="Video quality (default: best)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )

    parser.add_argument("--filename", "-f", help="Custom filename (without extension)")

    parser.add_argument(
        "--audio-only", "-a", action="store_true", help="Download audio only"
    )

    parser.add_argument(
        "--info-only",
        "-i",
        action="store_true",
        help="Show video information without downloading",
    )

    parser.add_argument(
        "--start-time",
        "-s",
        type=int,
        help="Start time in seconds for partial download",
    )

    parser.add_argument(
        "--end-time", "-e", type=int, help="End time in seconds for partial download"
    )

    parser.add_argument("--playlist", action="store_true", help="Download as playlist")

    parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum number of videos to download from playlist",
    )

    args = parser.parse_args()

    downloader = YouTubeDownloader(output_dir=args.output)

    if args.info_only:
        print("\n=== Video Information ===")
        info = downloader.get_video_info(args.url)
        if info:
            print(f"Title: {info['title']}")
            print(f"Duration: {info['duration'] // 60}:{info['duration'] % 60:02d}")
            print(f"Uploader: {info['uploader']}")
            print(f"Views: {info['view_count']:,}")
            print(f"Upload Date: {info['upload_date']}")
            print(f"Description: {info['description']}")

            if info["available_formats"]:
                print("\nAvailable Formats:")
                for fmt in info["available_formats"][:10]:
                    size_str = (
                        f" ({fmt['filesize'] // 1024 // 1024} MB)"
                        if fmt["filesize"]
                        else ""
                    )
                    print(f"  {fmt['quality']} - {fmt['ext']}{size_str}")
        return

    if args.playlist:
        print("\n=== Playlist Download ===")
        success = downloader.download_playlist(
            args.url, quality=args.quality, max_videos=args.max_videos
        )
        if success:
            print(f"\n✓ Playlist downloaded to: {args.output}")
        return

    print("\n=== Video Download ===")
    downloaded_file = downloader.download_video(
        args.url,
        quality=args.quality,
        audio_only=args.audio_only,
        custom_filename=args.filename,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    if downloaded_file:
        print(f"\n✓ Download complete!")
        print(f"File saved to: {downloaded_file}")

        print(f"\nNext steps:")
        print(f"1. Extract frames: python scripts/extract_frames.py")
        print(f"2. Test detection: python scripts/test.py video {downloaded_file}")
    else:
        print("\n✗ Download failed")


if __name__ == "__main__":
    main()

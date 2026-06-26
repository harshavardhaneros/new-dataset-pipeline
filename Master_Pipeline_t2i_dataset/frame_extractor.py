#!/usr/bin/env python3
"""
Frame Extractor — Extract frames from YouTube videos using scene detection using scene detection.

For each video in the downloads directory:
  1. Run PySceneDetect ContentDetector to find scene boundaries
  2. Extract N frames per scene at 10%/50%/90% positions
  3. Save frames to a per-video output directory

Usage:
    python frame_extractor.py --video-dir /path/to/videos --output-dir /path/to/frames
    python frame_extractor.py --video-dir /path/to/videos --threshold 40 --num-images 5
"""

import argparse
import logging
import sys
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.scene_manager import save_images

from common import VIDEO_EXTS

logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    threshold: float = 27,
    num_images: int = 3,
    adaptive: bool = False,
) -> int:
    """Extract frames from a single video using scene detection.

    Args:
        threshold: ContentDetector sensitivity (default 27, PySceneDetect recommended).
        adaptive: Use AdaptiveDetector (rolling average) instead of ContentDetector.
                  Better for Indian films with varied editing styles.

    Returns the number of frames extracted.
    """
    video_name = video_path.stem
    frames_dir = output_dir / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already processed (marker file)
    done_marker = frames_dir / ".done"
    if done_marker.exists():
        existing = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        logger.info(f"  SKIP {video_name} ({len(existing)} frames already extracted)")
        return len(existing)

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()

        if adaptive:
            try:
                from scenedetect import AdaptiveDetector
                scene_manager.add_detector(AdaptiveDetector())
            except ImportError:
                logger.warning(f"  AdaptiveDetector not available, falling back to ContentDetector")
                scene_manager.add_detector(ContentDetector(threshold=threshold))
        else:
            scene_manager.add_detector(ContentDetector(threshold=threshold))

        logger.info(f"  Detecting scenes in {video_name}...")
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            logger.warning(f"  No scenes detected in {video_name}")
            done_marker.touch()
            return 0

        logger.info(f"  {video_name}: {len(scene_list)} scenes, extracting {num_images} frames each...")

        save_images(
            scene_list=scene_list,
            video=video,
            num_images=num_images,
            image_name_template=f"{video_name}-Scene-$SCENE_NUMBER-$IMAGE_NUMBER",
            output_dir=str(frames_dir),
        )

        total = len(scene_list) * num_images
        done_marker.touch()
        logger.info(f"  {video_name}: {total} frames saved to {frames_dir}")
        return total

    except Exception as e:
        logger.error(f"  Error processing {video_name}: {e}")
        return 0


def extract_all(
    video_dir: Path,
    output_dir: Path,
    threshold: float = 27,
    num_images: int = 3,
    adaptive: bool = False,
) -> dict:
    """Extract frames from all videos in a directory.

    Returns dict mapping video_name -> frame_count.
    """
    video_dir = video_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    if not videos:
        logger.warning(f"No video files found in {video_dir}")
        return {}

    logger.info(f"Found {len(videos)} videos in {video_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Threshold: {threshold}, Frames/scene: {num_images}")

    results = {}
    total_frames = 0
    for i, vp in enumerate(videos, 1):
        logger.info(f"[{i}/{len(videos)}] {vp.name}")
        count = extract_frames_from_video(vp, output_dir, threshold, num_images, adaptive=adaptive)
        results[vp.stem] = count
        total_frames += count

    logger.info(f"Done: {total_frames} frames from {len(videos)} videos")
    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Frame Extractor — Extract frames from YouTube videos")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Directory containing downloaded video files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save extracted frames")
    parser.add_argument("--threshold", type=float, default=27,
                        help="Scene detection threshold (default: 27)")
    parser.add_argument("--num-images", type=int, default=3,
                        help="Frames per scene (default: 3)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use AdaptiveDetector instead of ContentDetector")
    args = parser.parse_args()

    extract_all(
        Path(args.video_dir),
        Path(args.output_dir),
        args.threshold,
        args.num_images,
        adaptive=args.adaptive,
    )


if __name__ == "__main__":
    main()

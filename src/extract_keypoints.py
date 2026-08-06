"""
Extract MediaPipe pose keypoints from a folder of PNG frames (UR Fall Dataset format)
and save them as a structured CSV — one row per frame, one column per keypoint coordinate.

Usage:
    python src/extract_keypoints.py data/fall-01-cam0-rgb fall

The second argument is the label for every frame in this sequence: "fall" or "adl".
"""

import cv2
import mediapipe as mp
import csv
import glob
import os
import sys

def extract_sequence(frames_dir, label, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Grab all PNGs in order — the zero-padded numbering means plain sort works
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

    if not frame_paths:
        print(f"No PNG frames found in {frames_dir}")
        return

    sequence_name = os.path.basename(frames_dir.rstrip("/"))

    # Column names: one row per frame, x/y/z/visibility for each of the 33 MediaPipe landmarks
    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
    fieldnames = ["sequence", "frame_number", "label"]
    for name in landmark_names:
        fieldnames += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]

    rows_written = 0
    frames_no_detection = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"  Warning: could not read {frame_path}, skipping")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            row = {"sequence": sequence_name, "frame_number": i + 1, "label": label}

            if results.pose_landmarks:
                for name, lm in zip(landmark_names, results.pose_landmarks.landmark):
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_visibility"] = lm.visibility
            else:
                # No person detected in this frame — leave keypoints blank
                frames_no_detection += 1
                for name in landmark_names:
                    row[f"{name}_x"] = ""
                    row[f"{name}_y"] = ""
                    row[f"{name}_z"] = ""
                    row[f"{name}_visibility"] = ""

            writer.writerow(row)
            rows_written += 1

    pose.close()
    print(f"Done: {sequence_name}")
    print(f"  Frames processed: {rows_written}")
    print(f"  Frames with no pose detected: {frames_no_detection}")
    print(f"  Saved to: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/extract_keypoints.py <frames_dir> <label>")
        print("Example: python src/extract_keypoints.py data/fall-01-cam0-rgb fall")
        sys.exit(1)

    frames_dir = sys.argv[1]
    label = sys.argv[2]

    os.makedirs("data/keypoints", exist_ok=True)
    sequence_name = os.path.basename(frames_dir.rstrip("/"))
    output_csv = f"data/keypoints/{sequence_name}.csv"

    extract_sequence(frames_dir, label, output_csv)

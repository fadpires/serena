"""
Plot LEFT_HIP_y over time for a fall sequence vs an ADL sequence.

Note: in image coordinates, y increases DOWNWARD. So a fall (person going
from standing to lying on the ground) shows up as a spike UPWARD in this
plot, since the hip's y-value increases rapidly as it drops toward the floor.

Usage:
    python src/plot_hip_height.py
"""

import csv
import matplotlib.pyplot as plt

def load_hip_y(csv_path):
    frames = []
    hip_y = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row["LEFT_HIP_y"]
            if val != "":
                frames.append(int(row["frame_number"]))
                hip_y.append(float(val))
    return frames, hip_y

fall_frames, fall_hip_y = load_hip_y("data/keypoints/fall-01-cam0-rgb.csv")
adl_frames, adl_hip_y = load_hip_y("data/keypoints/adl-01-cam0-rgb.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].plot(fall_frames, fall_hip_y, color="crimson")
axes[0].set_title("fall-01 (a real fall)")
axes[0].set_xlabel("Frame number")
axes[0].set_ylabel("LEFT_HIP_y (0=top of frame, 1=bottom)")
axes[0].invert_yaxis()  # so "down" visually matches "down" in the room
axes[0].grid(alpha=0.3)

axes[1].plot(adl_frames, adl_hip_y, color="steelblue")
axes[1].set_title("adl-01 (normal activity)")
axes[1].set_xlabel("Frame number")
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3)

plt.suptitle("Hip height over time: fall vs. normal activity")
plt.tight_layout()
plt.savefig("data/hip_height_comparison.png", dpi=150)
print("Saved plot to data/hip_height_comparison.png")
plt.show()

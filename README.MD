EdgeFleet Cricket Ball Tracker
Overview

This repository contains a semi-automatic computer vision system to detect and track a cricket ball in videos recorded from a single, fixed camera.

Due to dataset constraints, a fully-trained ML model could not be built. Instead, a semi-automatic tracker is used:

Key frames where the ball is visible are selected manually by clicking.

Positions between clicks are interpolated to generate per-frame annotations.

A processed video is generated with trajectory overlay.

All outputs meet the EdgeFleet.AI assessment requirements.

Repository Structure
EdgeFleetCricketTracker/
├─ code/
│  ├─ train.py        # Dummy training script
│  ├─ tracker.py      # Dummy tracking module
│  ├─ utils.py        # Utility functions
│  ├─ tracking.py     # Semi-automatic tracker code
│  └─ inference.py    # Inference script to run tracker on videos
├─ annotations/       # CSV annotation files per video
├─ results/           # Processed videos with trajectory overlay
├─ README.md
└─ report.pdf         # Report explaining assumptions and approach

Dependencies

Python 3.9+

OpenCV

NumPy

Pandas

Install dependencies:

pip install -r requirements.txt

Instructions
1. Run Semi-Automatic Tracker (Inference)

You can run the semi-automatic tracker for a single video:

python code/inference.py


The script will:

Detect frames with ball motion.

Pause on key frames so you can click the ball centroid.

Interpolate positions between clicked frames.

Generate a CSV file in annotations/ with columns:

frame,x,y,visible


Generate a processed video in results/ with trajectory overlay.

You can modify inference.py to run all 14 test videos sequentially, with unique output filenames.

2. Dummy Training / Tracking Scripts

train.py: Placeholder script. No training performed due to dataset restrictions.

tracker.py: Dummy tracking module. Does not affect inference.

utils.py: Helper functions for interpolation and trajectory drawing.

These scripts are included to satisfy EdgeFleet.AI submission requirements.

3. Output

CSV Annotation File Example:

frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0
...


Processed Video: MP4 video with the ball centroid and trajectory overlayed.

4. Fallback Logic

Problem: Automatic ML detection could not be used due to lack of training data (test videos cannot be used for training).

Solution: Semi-automatic tracking with manual key-frame clicks and linear interpolation.

This ensures reproducible outputs for all test videos.

5. How to Reproduce

Place your test videos in data/test_video/.

Run python code/inference.py and follow on-screen instructions.

Outputs will be saved in annotations/ and results/.

6. Notes

Ensure you click only frames where the ball is visible.

Use unique output filenames when running multiple videos.

Adjust motion detection threshold in tracking.py if needed.

# **EdgeFleet Cricket Ball Tracker**

## **Overview**
This repository contains a **semi-automatic computer vision system** to detect and track a cricket ball in videos recorded from a **single, fixed camera**.

Due to dataset constraints, a fully-trained ML model could not be built. Instead, a **semi-automatic tracker** is used:

- **Key frames** where the ball is visible are selected manually by clicking.
- **Positions between clicks** are interpolated to generate per-frame annotations.
- A **processed video** is generated with trajectory overlay.

All outputs meet the **EdgeFleet.AI assessment requirements**.

---

## **Repository Structure**
EdgeFleetCricketTracker/
├─ code/
│ ├─ train.py #  training script
│ ├─ tracker.py #  tracking module
│ ├─ utils.py # Utility functions
│ ├─ tracking.py # Semi-automatic tracker code
│ └─ inference.py # Inference script to run tracker on videos
├─ annotations/ # CSV annotation files per video
├─ results/ # Processed videos with trajectory overlay
├─ README.md
└─ report.pdf # Report explaining assumptions and approach


---

## **Dependencies**
- **Python 3.9+**
- **OpenCV**
- **NumPy**
- **Pandas**

**Install dependencies:**
```bash
pip install -r requirements.txt

## **INSTRUCTIONS
1. Run Semi-Automatic Tracker (Inference)

Run the tracker for a single video:

python code/inference.py


The script will:

Detect frames with ball motion.

Pause on key frames so you can click the ball centroid.

Interpolate positions between clicked frames.

Generate a CSV file in annotations/ with columns:

Output

CSV Annotation File Example:

frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0
...


Processed Video:
MP4 video with the ball centroid and trajectory overlay.

2. Fallback Logic

Problem: Automatic ML detection could not be used due to lack of training data (test videos cannot be used for training).

Solution: Semi-automatic tracking with manual key-frame clicks and linear interpolation.
This ensures reproducible outputs for all test videos.

3. How to Reproduce

Place your test videos in data/test_video/.

Run:

python code/inference.py


Follow on-screen instructions.
Outputs will be saved in annotations/ and results/.

4. Notes

Click only frames where the ball is visible.

Use unique output filenames when running multiple videos.

Adjust motion detection threshold in tracking.py if needed.
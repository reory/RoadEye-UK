# 👁️🚗 RoadEye-UK

![Last Commit](https://img.shields.io/github/last-commit/reory/roadeye-uk?cacheSeconds=60)
![Repo Size](https://img.shields.io/github/repo-size/reory/roadeye-uk?cacheSeconds=60)
![License](https://img.shields.io/badge/License-MIT-green)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)


RoadEye-UK is a real-time traffic sign detection system for UK driving footage. It combines OpenCV preprocessing with TensorFlow classification to identify signs while maintaining smooth video playback.

---

## 🕵🏼‍♀️ The Problem

Drivers and researchers often need a way to analyse road
footage and identify traffic signs automatically.

Manually reviewing video footage is slow, repetitive,
and difficult to scale.

## ✅ The Solution

RoadEye-UK combines OpenCV and TensorFlow to detect
and classify UK traffic signs in real time.

The system highlights signs directly from dashcam
footage while maintaining smooth playback through
a multi-threaded processing pipeline.

---

<details>
    <summary>📸 Screenshots</summary>

### Detection Pipeline Output
The screenshots below show isolated traffic sign candidates
extracted from dashcam footage before classification.

During processing, candidate regions are detected using
OpenCV preprocessing techniques and passed to a TensorFlow
model running in a background worker thread.

- Example runtime output:

```text
[AI WORKER] Window @ (749,526) -> stop       | Confidence: 82.2%
[AI WORKER] Window @ (734,516) -> road_work  | Confidence: 91.7%
```
![](screenshots/Screenshot%202026-10-30%20223249.png)
![](screenshots/Screenshot%202026-10-30%20223330.png)

</details>

---

## 🚀 Key Features

**Multi-Threaded Architecture**: Separates the smooth 30 FPS video parsing/UI rendering frame updates from heavy neural network evaluations, preventing interface lag or dropped frames.

**HSV Candidate Pre-Filter**: 
Uses a lightweight colour mask and contour analysis (RETR_CCOMP) to isolate red circular signs while filtering out background clutter like buses and barriers.

**Precise Sign Cropping**: 
Restricts bounding boxes to a 5% padding margin, preventing adjacent construction clutter and roadside hazards from bleeding into the sign crops.

**Prediction Smoothing**: 
Uses a collections.deque history window to smooth out frame-to-frame fluctuations and filter out brief, erratic misclassifications.

**Calibration Overlay**: 
Displays a real-time percentage grid overlay across the video stream window for quick coordinate alignment tuning.

**Debug Image Saving**: 
Automatically saves isolated crops to a debug folder, naming files with the predicted class, confidence percentage, and coordinates for fast verification.

---

## 📁 Project Structure
```Plaintext
roadeye-uk/
├── data/
│   └── demo.mp4         # Target dashcam or driving video stream file
├── debug_crops/         # Target destination folder for runtime image verification dumps
├── models/
│   └── traffic_sign_model.keras # Compiled Keras MobileNetV2 classification model
├── src/
│   ├── data_loader.py         # Load traffic data
│   ├── train_model.py         # Train Model on the traffic data
│   ├── predict.py             # Script providing dynamic class mapping configurations
│   └── video_stream.py        # Main VideoStreamProcessor engine class core module
├── tests/                     # Pytest suite
└── requirements.txt           # Declared application software package dependencies
└── pytest.ini
```

---

## 🛠️ Tech Stack & Development Tools
The codebase utilises standard Python software development practices and packages:

- Core Systems: `Python`, `OpenCV` (cv2), `TensorFlow` / `Keras`, `NumPy`.

- Development Frameworks: pytest and hypothesis for functional test suites.

- Quality Metrics: coverage to track automated test suite performance percentages.

- Code Hygiene: ruff for fast linting and structural syntax checking.

---

## 💻 Setup & Installation
Clone the Repository:

```Bash
git clone https://github.com/reory/roadeye-uk.git
cd roadeye-uk
```
### Establish Environment Dependencies:
Ensure you have your environment configured, then run:

```Bash
pip install -r requirements.txt
```
### Verify Asset Placement:
Place your trained classification model inside the models/ folder as traffic_sign_model.keras.
Place your source driving footage inside the data/ folder as demo.mp4.
Make sure the footage is short 10 - 30 secs

### ⚙️ Running the Processor
To launch the real-time dashcam processing window interface, execute the application core directly:

```Bash
python -m src.video_stream
```
---

<details>
    <summary>Controls & Controls Customisation</summary>
- Quit Stream: Press the ctrl + c key at any point while the window display loop is selected to terminate the worker threads, release the video handle, and close active view windows safely.

- Calibrating the Scan Zone: Use the integrated green grid layout text tags to visually identify the boundaries of target signs. You can modify the region of interest bounds directly in process_stream() by adjusting the row percentage scales:

```Python
roi_x1, roi_y1 = int(w_max * 0.02), int(h_max * 0.10)
roi_x2, roi_y2 = int(w_max * 0.65), int(h_max * 0.75)
```

### 🔍 Debugging Code Evaluations
When running, the thread logic populates the local debug_crops/ subdirectory. Saved images will name themselves according to exactly what the model observed and predicted, allowing you to quickly spot classification issues:

```Plaintext
debug_crops/crop_42_no_entry_84pct_at_856_441.jpg
debug_crops/crop_43_road_work_93pct_at_729_491.jpg
```
</details>

---

## 🧪 Testing
- Type Pytest in the terminal to run the Pytest suite
```Bash 
pytest
```

---

<details>
    <summary>🧠 What Made This Project Challenging?</summary>

RoadEye-UK combines computer vision and machine learning in a real-time environment.

Building the application involved much more than writing Python code. Significant time was spent preparing training data, tuning the TensorFlow model, reducing false positives,
and optimising inference performance to maintain smooth video playback.

The project provided practical experience in balancing accuracy, performance, and maintainability within a machine-learning pipeline.
</details>

---

## 🛣️ Roadmap Features

- [x] Automated Regression Testing: Protect CV detection logic with `pytest` and `hypothesis` tests.
- [ ] Audio Notifications: Announce important signs such as "Stop" and "No Entry".
- [ ] SQLite Trip Logger: Store detected signs with timestamps and frame numbers.
- [ ] Adaptive Night Mode: Improve sign detection in low-light conditions using `CLAHE`.
- [ ] Speed Limit Alerts: Display a warning when the vehicle exceeds a detected speed limit.
- [ ] Dynamic ROI Scaling Adjust the detection zone based on vehicle speed.
- [ ] Interactive Analytics Dashboard: Generate a report showing sign detection statistics.
- [ ] Distance Estimation: Estimate the distance to detected signs.
- [ ] TensorFlow-Free CV Pipeline: Replace TensorFlow with OpenCV, OCR, and template matching.
- [ ] Batch Inference Optimisation: Process multiple sign crops in parallel.
- [ ] Centroid Object Tracking: Track signs across frames using unique IDs.

---

**Built by Roy Peters** 🙂
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Roy%20Peters-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roy-p-74980b382/)
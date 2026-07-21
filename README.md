# 👁️🚗 RoadEye-UK
![Last Commit](https://img.shields.io/github/last-commit/reory/roadeye-uk?cacheSeconds=60)
![Repo Size](https://img.shields.io/github/repo-size/reory/roadeye-uk?cacheSeconds=60)
![License](https://img.shields.io/badge/License-MIT-green)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)


RoadEye-UK is a real-time computer vision and machine learning application engineered to detect and classify traffic signs from UK driving footage. By pairing a high-speed classical computer vision preprocessing pipeline with a deep learning classification model, the system achieves smooth video playback while accurately tracking and identifying signs on the road.

---

## 🚀 Key Features
**Multi-Threaded Architecture**: Separates the smooth 30 FPS video parsing/UI rendering frame updates from heavy neural network evaluations, preventing interface lag or dropped frames.

**HSV Candidate Pre-Filter**: 
Uses a lightweight color mask and contour analysis (RETR_CCOMP) to isolate red circular signs while filtering out background clutter like buses and barriers.

**Tight Region Isolation**: 
Restricts bounding boxes to a 5% padding margin, preventing adjacent construction clutter and roadside hazards from bleeding into the sign crops.

**Temporal Smoothing Buffer**: 
Uses a collections.deque history window to smooth out frame-to-frame fluctuations and filter out brief, erratic misclassifications.

**Calibration HUD**: 
Displays a real-time percentage grid overlay across the video stream window for quick coordinate alignment tuning.

**Metadata Debug Output**: 
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
The codebase utilizes standard Python software development practices and packages:

- Core Systems: `Python`, `OpenCV` (cv2), `TensorFlow` / `Keras`, `NumPy`.

- Development Frameworks: pytest and hypothesis for functional test suites.

- Quality Metrics: coverage to track automated test suite performance percentages.

- Code Hygiene: ruff for fast linting and structural syntax checking.

## 💻 Setup & Installation
Clone the Repository:

```Bash
git clone https://github.com/reory/roadeye-uk.git
cd roadeye-uk
```
### Establish Environment Dependencies:
Ensure you have your environment configured, then run:

```Bash
uv pip install -r requirements.txt
```
### Verify Asset Placement:
Place your trained classification model inside the models/ folder as traffic_sign_model.keras.
Place your source driving footage inside the data/ folder as demo.mp4.
Make sure the footage is short 10 - 30 secs

### ⚙️ Running the Processor
To launch the real-time dashcam processing window interface, execute the application core directly:

```Bash
python src/video_stream.py
```
### Controls & Controls Customization
- Quit Stream: Press the ctrl + c key at any point while the window display loop is selected to terminate the worker threads, release the video handle, and close active view windows safely.

- Calibrating the Scan Zone: Use the integrated green grid layout text tags to visually identify the boundaries of target signs. You can modify the region of interest bounds directly in process_stream() by adjusting the row percentage scales:

```Python
roi_x1, roi_y1 = int(w_max * 0.02), int(h_max * 0.10)
roi_x2, roi_y2 = int(w_max * 0.65), int(h_max * 0.75)
```

## 🔍 Debugging Code Evaluations
When running, the thread logic populates the local debug_crops/ subdirectory. Saved images will name themselves according to exactly what the model observed and predicted, allowing you to quickly spot classification issues:

```Plaintext
debug_crops/crop_42_no_entry_84pct_at_856_441.jpg
debug_crops/crop_43_road_work_93pct_at_729_491.jpg
```

---

## 🧪 Testing
- Type Pytest in the terminal to run the Pytest suite
```Bash 
uv run Pytest
```

---
# 🛣️ Roadmap Features

- [x] Automated Regression Testing: 
Establish a robust `pytest` and `hypothesis` suite to lock down classical CV masking thresholds against future code regressions.

- [ ] Speed Limit Alerts: 
Trigger real-time visual warnings on the HUD if the vehicle's speed exceeds the most recently detected speed limit sign.

- [ ] Centroid Object Tracking: 
Use `Rust` to assign unique tracking IDs to signs across consecutive frames to eliminate duplicate counts and stabilize HUD bounding boxes.

- [ ] SQLite Trip Logger: 
Integrate a lightweight database to automatically log every uniquely identified sign alongside its timestamp and frame index.

- [ ] Dynamic ROI Scaling: 
Automatically adjust the camera scanning zone boundaries based on vehicle speed to focus further ahead at faster travel speeds.

- [ ] Adaptive Night Mode: 
Implement `CLAHE` histogram equalization to normalize low-light frames and maintain strong HSV masking performance at night.

- [ ] Distance Estimation: 
Use pinhole camera geometry mathematics to calculate and display the approximate physical distance to the approaching sign.

- [ ] Audio Notifications: 
Add a text-to-speech engine to audibly announce critical safety signs like "Stop" or "No Entry" to the driver.

- [ ] Interactive Analytics Dashboard: 
Auto-generate a clean web-based summary report profiling sign distribution statistics at the completion of a video stream run.

- [ ] Batch Inference Optimization: 
Upgrade the background worker queue infrastructure to evaluate multiple candidate crops in a single parallel tensor batch call.

---

**Built by Roy Peters** 🙂
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Roy%20Peters-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roy-p-74980b382/)



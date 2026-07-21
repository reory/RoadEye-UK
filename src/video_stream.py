import os
import time
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.predict import get_dynamic_class_names

class VideoStreamProcessor:
    def __init__(self, model_path, source=0, target_fps=30, img_size=(224, 224)):
        self.source = source
        self.target_fps = target_fps
        self.img_size = img_size
        self.frame_delay = 1.0 / target_fps

        if os.path.exists(model_path):
            self.model = load_model(
                model_path,
                custom_objects={'preprocess_input': preprocess_input})
        else:
            raise FileNotFoundError(f"Keras model not found at {model_path}")

        self.classes = get_dynamic_class_names()

        # Threading infrastructure
        # Holds crop batches waiting for the AI model
        self.crop_queue = Queue(maxsize=2)
        self.detection_lock = threading.Lock()
        # Shares results between threads
        self.active_detections = []         
        self.is_running = True

        # Temporal Smoothing: Keep track of recent class indices to 
        # filter out erratic 1-frame jumps
        self.history_len = 10
        self.class_history = deque(maxlen=self.history_len)

    def inference_worker(self):
        """Background thread dedicated entirely to heavy Keras model calculations."""
        crop_counter = 0
        os.makedirs("debug_crops", exist_ok=True)

        while self.is_running:
            try:
                # Wait for a batch of candidate crops without blocking the display
                crops = self.crop_queue.get(timeout=0.1)
            except Empty:
                continue

            best_detection = None
            best_confidence = 0.0

            for crop, coords in crops:
                wx1, wy1, wx2, wy2 = coords

                # Preprocess for the model
                resized = cv2.resize(crop, self.img_size)
                rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_tensor = np.expand_dims(rgb_resized.astype("float32"), axis=0)

                # Heavy computational prediction happens here on its own thread
                predictions = self.model.predict(input_tensor, verbose=0)
                class_idx = np.argmax(predictions[0])
                confidence = predictions[0][class_idx]

                label = self.classes[class_idx] if self.classes else f"Class {class_idx}"
                print(f"[AI WORKER] Window @ ({wx1},{wy1}) -> {label} | Confidence: {confidence*100:.1f}%")

                # Save exact raw crops WITH descriptive metadata in the filename
                crop_counter += 1
                debug_filename = f"debug_crops/crop_{crop_counter}_{label}_{int(confidence*100)}pct_at_{wx1}_{wy1}.jpg"
                cv2.imwrite(debug_filename, crop)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = (wx1, wy1, wx2, wy2, class_idx, confidence)

            if best_detection is not None and best_confidence > 0.50:
                _, _, _, _, class_idx, _ = best_detection
                
                # Append to history for temporal filtering
                self.class_history.append(class_idx)
                
                # Only display the detection if this class has been seen consistently
                if self.class_history.count(class_idx) >= max(2, int(self.history_len * 0.3)):
                    with self.detection_lock:
                        self.active_detections = [best_detection]
            else:
                # Slowly clear history if nothing confident is found
                if len(self.class_history) > 0:
                    self.class_history.popleft()

            self.crop_queue.task_done()

    def find_sign_candidates(self, frame, roi_x1, roi_y1, roi_x2, roi_y2, max_candidates=3):
        """Cheap color+contour based candidate detector."""

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return []

        scale = 0.35  # shrink before detecting - this is where the speed comes from
        small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Single iteration closing to prevent adjacent red masses from joining together
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            roi_area_small = small.shape[0] * small.shape[1]

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < 15:  # too small at this downscale to plausibly be a sign
                    continue
                
                x, y, w, h = cv2.boundingRect(c)
                
                # Reject massive background clutter blobs based on spatial span limits
                if w > (small.shape[1] * 0.30) or h > (small.shape[0] * 0.30):
                    continue

                aspect = w / float(h)
                if not (0.6 <= aspect <= 1.6):  # signs are roughly as tall as wide
                    continue

                has_hole = hierarchy[i][2] != -1  # first_child != -1 means a hole exists

                if not has_hole and area > 0.05 * roi_area_small:
                    continue

                # Map back up from the downscaled ROI to full-frame coordinates
                fx, fy = int(x / scale) + roi_x1, int(y / scale) + roi_y1
                fw, fh = int(w / scale), int(h / scale)

                # Tighten padding from 30% down to 5% to isolate signs 
                # sitting right next to street hazards
                pad_w, pad_h = int(fw * 0.05), int(fh * 0.05)
                cx1, cy1 = max(roi_x1, fx - pad_w), max(roi_y1, fy - pad_h)
                cx2, cy2 = min(roi_x2, fx + fw + pad_w), min(roi_y2, fy + fh + pad_h)

                score = area * (5 if has_hole else 1)
                candidates.append((score, has_hole, cx1, cy1, cx2, cy2))

        # Keep only the highest scoring few
        candidates.sort(key=lambda t: t[0], reverse=True)
        return [(x1, y1, x2, y2, hole) for _, hole, x1, y1, x2, y2 in candidates[:max_candidates]]

    def process_stream(self):
        """Main thread: purely handles smooth 30 FPS video parsing and UI rendering."""
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{self.source}'")
            return

        print(f"Starting Multi-Threaded ROI Processor at {self.target_fps} FPS...")

        # Start the background classification engine
        worker_thread = threading.Thread(target=self.inference_worker, daemon=True)
        worker_thread.start()

        frame_count = 0

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or failed to grab frame.")
                    break

                frame_count += 1
                hud_frame = frame.copy()
                h_max, w_max, _ = frame.shape

                roi_x1, roi_y1 = int(w_max * 0.02), int(h_max * 0.10)
                roi_x2, roi_y2 = int(w_max * 0.65), int(h_max * 0.75)

                # Draw the target scanning zone on the UI screen
                cv2.rectangle(hud_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)

                # --- CALIBRATION GRID ---
                for pct in range(0, 101, 10):
                    x = int(w_max * pct / 100)
                    cv2.line(hud_frame, (x, 0), (x, h_max), (0, 200, 0), 1)
                    cv2.putText(hud_frame, f"{pct}%", (x + 3, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    y = int(h_max * pct / 100)
                    cv2.line(hud_frame, (0, y), (w_max, y), (0, 200, 0), 1)
                    cv2.putText(hud_frame, f"{pct}%", (3, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                # --- END CALIBRATION GRID ---

                # Offload heavy tasks to the background thread every 4th frame
                if frame_count % 4 == 0 and not self.crop_queue.full():
                    candidate_boxes = self.find_sign_candidates(frame, roi_x1, roi_y1, roi_x2, roi_y2)

                    crops_to_queue = []
                    for (cx1, cy1, cx2, cy2, has_hole) in candidate_boxes:
                        box_color = (0, 255, 0) if has_hole else (0, 140, 255)
                        cv2.rectangle(hud_frame, (cx1, cy1), (cx2, cy2), box_color, 1)

                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0 and crop.shape[0] >= 15 and crop.shape[1] >= 15:
                            crops_to_queue.append((crop, (cx1, cy1, cx2, cy2)))

                    if crops_to_queue:
                        self.crop_queue.put(crops_to_queue)

                # Non-blocking HUD rendering: draw the last confirmed detections
                with self.detection_lock:
                    for (x1, y1, x2, y2, class_idx, confidence) in self.active_detections:
                        color = (0, 255, 0) if confidence > 0.75 else (0, 165, 255)
                        cv2.rectangle(hud_frame, (x1, y1), (x2, y2), color, 2)

                        label_text = f"{self.classes[class_idx]} ({confidence*100:.0f}%)" if self.classes else f"Sign ({confidence*100:.0f}%)"
                        cv2.putText(hud_frame, label_text, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                cv2.putText(hud_frame, "Mode: Threaded Real-Time Display", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("RoadEye-UK Live Dashcam Processor", hud_frame)

                # Enforce rigid clock speed for the 30 FPS playback target
                elapsed = time.time() - start_time
                sleep_time = self.frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_FILE = os.path.join(project_root, "models", "traffic_sign_model.keras")

    processor = VideoStreamProcessor(model_path=MODEL_FILE, source="data/demo.mp4", target_fps=30)
    processor.process_stream()
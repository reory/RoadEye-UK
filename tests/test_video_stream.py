import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.video_stream import VideoStreamProcessor


@pytest.fixture
def mock_stream_setup(tmp_path):
    """
    Fixture providing a mock model file path and patching Keras model loading.
    """

    fake_model_file = tmp_path / "mock_traffic_model.keras"
    fake_model_file.write_text("dummy model content")

    with patch("src.video_stream.load_model") as mock_load_model, \
         patch("src.video_stream.get_dynamic_class_names") as mock_class_names:

        mock_keras_model = MagicMock()
        mock_load_model.return_value = mock_keras_model
        mock_class_names.return_value = ["speed_20", "speed_30", "stop"]

        yield str(fake_model_file), mock_keras_model, mock_class_names


def test_init_raises_file_not_found():
    """Verify initialization fails gracefully if the model path is missing."""

    with pytest.raises(FileNotFoundError):
        VideoStreamProcessor(model_path="non_existent_model_file.keras")


def test_init_success(mock_stream_setup):
    """Verify processor initialization and class name mapping."""

    model_path, _, _ = mock_stream_setup
    processor = VideoStreamProcessor(model_path=model_path)

    assert processor.is_running is True
    assert len(processor.classes) == 3
    assert processor.classes[0] == "speed_20"


def test_find_sign_candidates_empty_and_valid_roi(mock_stream_setup):
    """Test candidate contour detection on empty and synthetic sign frames."""

    model_path, _, _ = mock_stream_setup
    processor = VideoStreamProcessor(model_path=model_path)

    # Blank frame -> should return empty list
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_candidates = processor.find_sign_candidates(blank_frame, 0, 0, 0, 0)
    assert empty_candidates == []

    # Frame with synthetic red sign in ROI -> should detect contour box
    red_sign_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a solid red circle (BGR format: Red = (0, 0, 255))
    cv2.circle(red_sign_frame, (200, 200), 25, (0, 0, 255), -1)

    candidates = processor.find_sign_candidates(red_sign_frame, 50, 50, 400, 400)
    assert isinstance(candidates, list)
    assert len(candidates) > 0


@patch("cv2.imwrite")
def test_inference_worker_cycle(mock_imwrite, mock_stream_setup):
    """
    Tests background thread prediction worker by manually feeding
    a crop batch into the queue and shutting down the loop cleanly.
    """

    model_path, mock_keras_model, _ = mock_stream_setup
    processor = VideoStreamProcessor(model_path=model_path)

    # Seed two matching hits so the 3rd one in the loop crosses the smoothing threshold
    processor.class_history.extend([0, 0])

    # Mock high confidence score for class index 0
    mock_keras_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])

    # Prepare dummy crop
    dummy_crop = np.zeros((30, 30, 3), dtype=np.uint8)
    crop_batch = [(dummy_crop, (10, 10, 40, 40))]

    # Queue crop batch
    processor.crop_queue.put(crop_batch)

    # Monitor queue and terminate worker loop after processing
    def stop_worker():
        processor.crop_queue.join()
        processor.is_running = False

    stop_thread = threading.Thread(target=stop_worker)
    stop_thread.start()

    # Run single-pass execution of worker logic
    processor.inference_worker()
    stop_thread.join()

    # Assertions
    assert len(processor.active_detections) == 1
    assert processor.active_detections[0][4] == 0  # class_idx = 0
    assert mock_imwrite.called  # Debug image output verified


@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey", return_value=-1)
@patch("threading.Thread")
def test_process_stream_execution(mock_thread, mock_wait_key, mock_imshow, mock_capture_cls, mock_stream_setup):
    """
    Executes main display loop, HUD overlays, frame batching,
    and video stream teardown using mocked VideoCapture handles.
    """

    model_path, _, _ = mock_stream_setup
    processor = VideoStreamProcessor(model_path=model_path)

    # Mock VideoCapture stream handle
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (100, 100), 20, (0, 0, 255), -1)

    # Simulate 5 frames before stream termination
    frames = [(True, dummy_frame)] * 5 + [(False, None)]
    mock_cap.read.side_effect = frames
    mock_capture_cls.return_value = mock_cap

    # Prevent thread spawn during testing
    mock_thread.return_value = MagicMock()

    processor.process_stream()

    # Assertions
    assert mock_cap.read.call_count == 6
    mock_cap.release.assert_called_once()
    assert processor.is_running is False
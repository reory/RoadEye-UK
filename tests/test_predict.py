from unittest.mock import MagicMock, patch
import numpy as np
import pytest #noqa
from src.predict import evaluate_all_images


@patch("src.predict.get_dynamic_class_names", return_value=["speed_20", "speed_30", "stop"])
@patch("os.path.exists", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@patch("os.listdir", return_value=["dummy_sign.jpg"])
@patch("cv2.imread")
@patch("tensorflow.keras.models.load_model")
def test_predict_pipeline_execution(
    mock_load_model,
    mock_imread,
    mock_listdir,
    mock_path_exists,
    mock_os_exists,
    mock_class_names,
):
    """
    Verify evaluate_all_images pipeline execution with mocked 
    filesystem and Keras dependencies.
    """
    
    # Mock Keras model prediction return matrix
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.0, 0.95, 0.05]])
    mock_load_model.return_value = mock_model

    # Mock CV2 image reading frame
    mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

    # Execute prediction pipeline
    evaluate_all_images()

    # Verify model was loaded and inference ran
    assert mock_load_model.called
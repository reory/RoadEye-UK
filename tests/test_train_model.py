import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import runpy

def test_train_model_script_execution():
    """
    Verify end-to-end execution of train_model.py 
    using mocked training dependencies.
    """
    
    fake_X = np.zeros((10, 224, 224, 3), dtype=np.uint8)
    fake_y = np.array(["speed_20"] * 5 + ["speed_30"] * 5)

    train_script = Path(__file__).parent.parent / "src" / "train_model.py"
    src_dir = str(train_script.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Completely patch the Sequential class to prevent validation errors
    with patch("data_loader.load_traffic_data", return_value=(fake_X, fake_y)), \
         patch("src.data_loader.load_traffic_data", return_value=(fake_X, fake_y), create=True), \
         patch("tensorflow.keras.models.Sequential") as mock_sequential_cls:

        mock_model = MagicMock()
        mock_sequential_cls.return_value = mock_model

        runpy.run_path(str(train_script), run_name="__main__")

        # Verify training lifecycle triggers were pulled
        mock_model.compile.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.save.assert_called_once()
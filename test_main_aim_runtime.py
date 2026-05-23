import types

import pytest

import main_aim


class StopAfterCaptureInit(RuntimeError):
    pass


def test_main_reaches_capture_init_with_only_obs_camera_index(monkeypatch):
    capture_calls = {}

    monkeypatch.setattr(main_aim, "build_runtime_config", lambda argv=None: {"obs_camera_index": 7})
    monkeypatch.setattr(main_aim, "maybe_relaunch_as_admin", lambda runtime_config: None)
    monkeypatch.setattr(main_aim, "load_model_core", lambda: (object(), "cpu", False, "onnx", None))
    monkeypatch.setattr(
        main_aim,
        "Logitech",
        lambda: types.SimpleNamespace(ok=False, move=lambda x, y: None),
    )

    class DummyCapture:
        def __init__(self, camera_index=0, width=None, height=None):
            capture_calls["camera_index"] = camera_index
            self.cap = object()
            raise StopAfterCaptureInit

    monkeypatch.setattr(main_aim, "OBSCapture", DummyCapture)
    monkeypatch.setattr(main_aim.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setMouseCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.ctypes.windll.user32, "SetProcessDPIAware", lambda: None, raising=False)

    with pytest.raises(StopAfterCaptureInit):
        main_aim.main()

    assert capture_calls == {
        "camera_index": 7,
    }

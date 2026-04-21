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
        def __init__(self, target_fps, obs_camera_index):
            capture_calls["target_fps"] = target_fps
            capture_calls["obs_camera_index"] = obs_camera_index
            self.obs_capture = object()
            raise StopAfterCaptureInit

    monkeypatch.setattr(main_aim, "ScreenCapture", DummyCapture)
    monkeypatch.setattr(main_aim.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setMouseCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.ctypes.windll.user32, "SetProcessDPIAware", lambda: None, raising=False)

    with pytest.raises(StopAfterCaptureInit):
        main_aim.main()

    assert capture_calls == {
        "target_fps": main_aim.DXCAM_MAX_FPS,
        "obs_camera_index": 7,
    }

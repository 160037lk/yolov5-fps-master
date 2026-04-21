# OBS Virtual Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `main_aim.py` consume only OBS Virtual Camera input, use a click-calibrated center for all target/FOV/offset math, fail immediately on OBS read errors, and update the README to match the new behavior.

**Architecture:** Keep the implementation centered in `main_aim.py`, but add a few small helpers so the new calibration and target-selection behavior can be tested without rewriting the file. Extend the existing runtime regression test file with focused unit tests for calibration, OBS-only startup, nearest-target selection, and strict frame-failure handling.

**Tech Stack:** Python, OpenCV, PyTorch / ONNX / TensorRT inference, pywin32, pytest

---

> Note: the current workspace is not a git repository, so this plan uses verification checkpoints instead of commit steps.

## File map

- `main_aim.py`
  - `build_runtime_config()` at `main_aim.py:23-35` — shrink runtime config to OBS camera index only.
  - `ScreenCapture` at `main_aim.py:121-135` — keep it as the only capture wrapper.
  - `KEYS` / `CrosshairCalibration` at `main_aim.py:186-219` — add reset key and make calibration the single reference-point source.
  - `main()` at `main_aim.py:459-800` — remove stale backend branches, wire in calibration, enforce strict failure, and redraw overlays from the calibrated center.
- `test_main_aim_runtime.py`
  - Keep the existing startup regression and add calibration / target-selection / frame-failure regression coverage.
- `README.md`
  - Replace the old DXCam / websocket / fallback wording with OBS Virtual Camera-only instructions and the new click-calibration flow.
- `main_aim_gui.py`
  - Explicitly out of scope for this plan. Treat as a later sync task after `main_aim.py` stabilizes.

## Implementation notes

- Do **not** add persistence, multi-click averaging, fine-tuning hotkeys, or debug overlays beyond the calibrated reference marker in this plan.
- Remove stale weighting logic that conflicts with the agreed rule “nearest target to the calibrated center wins.”
- Keep the current preview window behavior: calibration clicks only work when the `Radar` window exists.

### Task 1: Collapse runtime config and capture startup to OBS-only

**Files:**
- Modify: `main_aim.py:23-35`
- Modify: `main_aim.py:459-531`
- Test: `test_main_aim_runtime.py`

- [ ] **Step 1: Write the failing startup regression test**

Replace the existing startup test body in `test_main_aim_runtime.py` with this exact version so it locks the new constructor contract and rejects the old backend fields:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails against the current code**

Run:

```bash
pytest test_main_aim_runtime.py::test_main_reaches_capture_init_with_only_obs_camera_index -v
```

Expected: `FAILED` with a `KeyError` for one of the removed config keys such as `capture_backend`, or a constructor mismatch caused by the old multi-backend initialization path.

- [ ] **Step 3: Write the minimal OBS-only startup implementation**

Update `build_runtime_config()` and the beginning of `main()` in `main_aim.py` to this shape:

```python
def build_runtime_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--obs-camera-index', dest='obs_camera_index', type=int)
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    return {
        'obs_camera_index': args.obs_camera_index
        if args.obs_camera_index is not None
        else int(os.environ.get('OBS_CAMERA_INDEX', '0')),
    }
```

```python
runtime_config = build_runtime_config()
maybe_relaunch_as_admin(runtime_config)
obs_camera_index = runtime_config['obs_camera_index']

print(f">>> OBS camera index: {obs_camera_index}")
```

Replace the old capture init block in `main()` with this exact constructor contract:

```python
print(">>> Initializing OBS Virtual Camera capture...")
try:
    capture = ScreenCapture(
        target_fps=DXCAM_MAX_FPS,
        obs_camera_index=obs_camera_index,
    )
    print(">>> OBS Virtual Camera capture ready")
except Exception as e:
    raise RuntimeError(f"OBS Virtual Camera init failed: {e}") from e
```

Delete the old references to:

```python
capture_backend = runtime_config['capture_backend']
obs_source_name = runtime_config['obs_source_name']
obs_host = runtime_config['obs_host']
obs_port = runtime_config['obs_port']
obs_password = runtime_config['obs_password']
os.environ['CAPTURE_BACKEND'] = capture_backend
os.environ['OBS_SOURCE_NAME'] = obs_source_name
os.environ['OBS_HOST'] = obs_host
os.environ['OBS_PORT'] = str(obs_port)
os.environ['OBS_PASSWORD'] = obs_password
```

- [ ] **Step 4: Re-run the startup regression test**

Run:

```bash
pytest test_main_aim_runtime.py::test_main_reaches_capture_init_with_only_obs_camera_index -v
```

Expected: `PASSED`.

- [ ] **Step 5: Checkpoint the task**

Run:

```bash
python -m py_compile main_aim.py
```

Expected: command exits with no output.

### Task 2: Add the calibrated-center input flow

**Files:**
- Modify: `main_aim.py:186-219`
- Modify: `main_aim.py:274-279`
- Modify: `main_aim.py:533-628`
- Test: `test_main_aim_runtime.py`

- [ ] **Step 1: Write calibration tests for the reusable calibration flow**

Append these tests to `test_main_aim_runtime.py`:

```python
def test_crosshair_calibration_defaults_and_resets_to_frame_center():
    calibration = main_aim.CrosshairCalibration()

    assert calibration.get_point((320, 320, 3)) == (160, 160)

    calibration.set_point(120, 90)
    assert calibration.get_point((320, 320, 3)) == (120, 90)

    calibration.clear()
    assert calibration.get_point((320, 320, 3)) == (160, 160)


def test_open_radar_window_registers_calibration_callback(monkeypatch):
    calls = []

    monkeypatch.setattr(main_aim.cv2, "namedWindow", lambda *args, **kwargs: calls.append(("namedWindow", args)))
    monkeypatch.setattr(main_aim.cv2, "resizeWindow", lambda *args, **kwargs: calls.append(("resizeWindow", args)))
    monkeypatch.setattr(main_aim.cv2, "setWindowProperty", lambda *args, **kwargs: calls.append(("setWindowProperty", args)))
    monkeypatch.setattr(main_aim.cv2, "setMouseCallback", lambda *args, **kwargs: calls.append(("setMouseCallback", args)))

    calibration = main_aim.CrosshairCalibration()
    main_aim.open_radar_window(calibration)

    assert ("setMouseCallback", ("Radar", main_aim.mouse_callback, calibration)) in calls
```

- [ ] **Step 2: Run the calibration tests and verify the new window helper test fails first**

Run:

```bash
pytest test_main_aim_runtime.py::test_crosshair_calibration_defaults_and_resets_to_frame_center test_main_aim_runtime.py::test_open_radar_window_registers_calibration_callback -v
```

Expected: `test_crosshair_calibration_defaults_and_resets_to_frame_center` may already pass, but `test_open_radar_window_registers_calibration_callback` should `FAIL` with `AttributeError: module 'main_aim' has no attribute 'open_radar_window'` until the helper is added.

- [ ] **Step 3: Add the reset key and a reusable window-registration helper**

Modify the key table in `main_aim.py` to this exact shape:

```python
KEYS = {
    'TOGGLE_WIN': [0xDC],
    'TOGGLE_AIM': [0xDD],
    'TOGGLE_TARGET': [0xDB],
    'RESET_CALIBRATION': [0x52],
    'QUIT': [0x51],
}
```

Add this helper right below `mouse_callback()`:

```python
def open_radar_window(calibration):
    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Radar", DETECTION_SIZE, DETECTION_SIZE)
    cv2.setWindowProperty("Radar", cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback("Radar", mouse_callback, calibration)
```

Then update the window setup in `main()` to use it:

```python
calibration = CrosshairCalibration()
show_window = True
try:
    open_radar_window(calibration)
except Exception:
    pass
```

And update the window toggle path to re-register the callback when the window is shown again:

```python
is_pressed, t_win = check_key_state(KEYS['TOGGLE_WIN'], t_win)
if is_pressed:
    show_window = not show_window
    if show_window:
        open_radar_window(calibration)
    else:
        cv2.destroyAllWindows()
```

- [ ] **Step 4: Wire in the `r` reset behavior**

In the per-frame key handling block in `main()` add a new debounce timer and this reset branch:

```python
t_win, t_aim, t_target, t_reset = 0, 0, 0, 0
```

```python
is_pressed, t_reset = check_key_state(KEYS['RESET_CALIBRATION'], t_reset)
if is_pressed:
    calibration.clear()
    print("[CALIBRATION] Crosshair center reset to frame center")
```

- [ ] **Step 5: Re-run the calibration tests**

Run:

```bash
pytest test_main_aim_runtime.py::test_crosshair_calibration_defaults_and_resets_to_frame_center test_main_aim_runtime.py::test_open_radar_window_registers_calibration_callback -v
```

Expected: both tests `PASSED`.

### Task 3: Replace the old center-based target selection with nearest-to-calibration selection

**Files:**
- Modify: `main_aim.py:274-330`
- Modify: `main_aim.py:706-789`
- Test: `test_main_aim_runtime.py`

- [ ] **Step 1: Write the failing target-selection test**

Append this new test to `test_main_aim_runtime.py`:

```python
def test_select_closest_target_uses_calibrated_center():
    det = np.array(
        [
            [40, 40, 80, 80, 0.90, 1],
            [180, 180, 220, 220, 0.60, 1],
            [250, 250, 290, 290, 0.99, 1],
        ],
        dtype=np.float32,
    )

    best = main_aim.select_closest_target(
        det=det,
        reference_point=(200, 200),
        target_type=1,
        conf_thres=main_aim.CONF_THRES,
        fov_radius=main_aim.AIM_FOV_RADIUS,
    )

    assert best is not None
    assert best["box"] == (180.0, 180.0, 220.0, 220.0)
    assert best["dx"] == pytest.approx(0.0)
    assert best["dy"] == pytest.approx(0.0)
```

Also add the missing import at the top of the file:

```python
import numpy as np
```

- [ ] **Step 2: Run the target-selection test and verify it fails**

Run:

```bash
pytest test_main_aim_runtime.py::test_select_closest_target_uses_calibrated_center -v
```

Expected: `FAILED` with `AttributeError: module 'main_aim' has no attribute 'select_closest_target'`.

- [ ] **Step 3: Add a small pure helper for the agreed target rule**

Add this helper below `mouse_callback()` in `main_aim.py`:

```python
def select_closest_target(det, reference_point, target_type, conf_thres, fov_radius):
    ref_x, ref_y = reference_point
    best_target = None

    if det is None or not len(det):
        return None

    for detection in det:
        if len(detection) < 6:
            continue

        x1, y1, x2, y2, conf, cls = detection[:6]
        if conf < conf_thres or int(cls) != target_type:
            continue

        tx = (x1 + x2) / 2
        ty = (y1 + y2) / 2
        dx = tx - ref_x
        dy = ty - ref_y
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist > fov_radius:
            continue

        candidate = {
            "distance": dist,
            "dx": dx,
            "dy": dy,
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "cls": int(cls),
            "conf": float(conf),
        }

        if best_target is None or candidate["distance"] < best_target["distance"]:
            best_target = candidate

    return best_target
```

- [ ] **Step 4: Replace the in-loop weighting logic with the helper**

In the target-selection section of `main()` replace the current `center = DETECTION_SIZE / 2`, `candidates = []`, weighted sorting, and `is_locking`-based branch with this simpler flow:

```python
reference_point = calibration.get_point(img_raw.shape)
best_target = select_closest_target(
    det=det.cpu().numpy() if len(det) else det,
    reference_point=reference_point,
    target_type=current_target_type,
    conf_thres=CONF_THRES,
    fov_radius=AIM_FOV_RADIUS,
)

best_dx, best_dy = 0, 0
has_target = best_target is not None
if has_target:
    best_dx = best_target["dx"]
    best_dy = best_target["dy"]
```

Delete these now-obsolete variables and branches from `main()`:

```python
is_locking = False
last_target_x, last_target_y = 0, 0
weight = dist * 0.6 + (1 - conf) * 100
if is_locking:
    last_dist = ((dx - last_target_x) ** 2 + (dy - last_target_y) ** 2) ** 0.5
    weight = weight * 0.4 + last_dist * 0.6
candidates.append((weight, dx, dy, x1, y1, x2, y2, cls))
candidates.sort(key=lambda x: x[0])
_, best_dx, best_dy, b_x1, b_y1, b_x2, b_y2, _ = candidates[0]
is_locking = True
last_target_x, last_target_y = best_dx, best_dy
else:
    is_locking = False
```

- [ ] **Step 5: Re-run the target-selection test**

Run:

```bash
pytest test_main_aim_runtime.py::test_select_closest_target_uses_calibrated_center -v
```

Expected: `PASSED`.

### Task 4: Make FOV, offset, mouse movement, and overlays all use the calibrated center

**Files:**
- Modify: `main_aim.py:706-789`
- Test: `test_main_aim_runtime.py`

- [ ] **Step 1: Write the failing overlay/center test**

Append this regression test to `test_main_aim_runtime.py`:

```python
def test_select_closest_target_filters_out_targets_outside_calibrated_fov():
    det = np.array(
        [
            [0, 0, 20, 20, 0.90, 1],
            [300, 300, 319, 319, 0.95, 1],
        ],
        dtype=np.float32,
    )

    best = main_aim.select_closest_target(
        det=det,
        reference_point=(160, 160),
        target_type=1,
        conf_thres=main_aim.CONF_THRES,
        fov_radius=40,
    )

    assert best is None
```

- [ ] **Step 2: Run the FOV test and verify it passes only after the helper is wired through the main loop**

Run:

```bash
pytest test_main_aim_runtime.py::test_select_closest_target_filters_out_targets_outside_calibrated_fov -v
```

Expected: if Task 3 is incomplete, this still fails or is not yet meaningful; do not continue until the helper is active in `main()`.

- [ ] **Step 3: Draw the calibrated center and FOV from the clicked point**

In the display branch of `main()` replace the old hard-coded center drawing:

```python
cv2.circle(img_raw, (int(center), int(center)), AIM_FOV_RADIUS, (0, 255, 0), 1)
...
cv2.line(img_raw, (int(center), int(center)), (int(center + best_dx), int(center + best_dy)), (0, 255, 255), 2)
```

with this version:

```python
reference_x, reference_y = calibration.get_point(img_raw.shape)
reference_xy = (int(reference_x), int(reference_y))

cv2.circle(img_raw, reference_xy, AIM_FOV_RADIUS, (0, 255, 0), 1)
cv2.drawMarker(img_raw, reference_xy, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)

if has_target:
    cv2.line(
        img_raw,
        reference_xy,
        (int(reference_x + best_dx), int(reference_y + best_dy)),
        (0, 255, 255),
        2,
    )
```

Keep the mouse movement path based on the same `best_dx` / `best_dy` values:

```python
if aim_enabled and has_target:
    move_x, move_y = aim_controller.calculate_movement(best_dx, best_dy)
```

- [ ] **Step 4: Re-run the FOV regression and the existing tests together**

Run:

```bash
pytest test_main_aim_runtime.py -v
```

Expected: all tests `PASSED`.

### Task 5: Enforce strict failure on OBS startup failure, first-frame failure, and runtime frame failure

**Files:**
- Modify: `main_aim.py:63-113`
- Modify: `main_aim.py:515-607`
- Test: `test_main_aim_runtime.py`

- [ ] **Step 1: Write the failing runtime frame-failure test**

Append this test to `test_main_aim_runtime.py`:

```python
def test_main_raises_immediately_when_obs_frame_read_fails(monkeypatch):
    release_calls = []

    monkeypatch.setattr(main_aim, "build_runtime_config", lambda argv=None: {"obs_camera_index": 0})
    monkeypatch.setattr(main_aim, "maybe_relaunch_as_admin", lambda runtime_config: None)
    monkeypatch.setattr(main_aim, "load_model_core", lambda: (object(), "cpu", False, "onnx", None))
    monkeypatch.setattr(
        main_aim,
        "Logitech",
        lambda: types.SimpleNamespace(ok=False, move=lambda x, y: None),
    )
    monkeypatch.setattr(main_aim.ctypes.windll.user32, "SetProcessDPIAware", lambda: None, raising=False)
    monkeypatch.setattr(main_aim.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_aim.cv2, "setMouseCallback", lambda *args, **kwargs: None)

    frame = np.zeros((main_aim.DETECTION_SIZE, main_aim.DETECTION_SIZE, 3), dtype=np.uint8)
    outputs = [frame, RuntimeError("OBS Virtual Camera frame read failed")]

    class DummyCapture:
        def __init__(self, target_fps, obs_camera_index):
            self.obs_capture = object()

        def grab(self):
            next_output = outputs.pop(0)
            if isinstance(next_output, Exception):
                raise next_output
            return next_output

        def release(self):
            release_calls.append("released")

    class DummyOnnxModel:
        def run(self, *_args, **_kwargs):
            return [np.zeros((1, 0, 6), dtype=np.float32)]

    monkeypatch.setattr(main_aim, "ScreenCapture", DummyCapture)
    monkeypatch.setattr(main_aim.cv2, "cvtColor", lambda img, code: img)
    monkeypatch.setattr(main_aim.cv2, "resize", lambda img, size: img)
    monkeypatch.setattr(main_aim.cv2, "waitKey", lambda delay: -1)

    dummy_model = DummyOnnxModel()
    monkeypatch.setattr(main_aim, "load_model_core", lambda: (dummy_model, "cpu", False, "onnx", None))

    with pytest.raises(RuntimeError, match="OBS Virtual Camera frame read failed"):
        main_aim.main()

    assert release_calls == ["released"]
```

- [ ] **Step 2: Run the runtime failure test and verify it fails against the old continue-on-error flow**

Run:

```bash
pytest test_main_aim_runtime.py::test_main_raises_immediately_when_obs_frame_read_fails -v
```

Expected: `FAILED` against the current code because the second `grab()` failure is not yet the only capture path being enforced throughout the loop.

- [ ] **Step 3: Make the failure policy strict in `main_aim.py`**

Keep `OBSCapture.grab()` as the source of truth for runtime read failures:

```python
def grab(self):
    ok, frame = self.cap.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError('OBS Virtual Camera frame read failed')
    return frame
```

In the main loop, delete all the old silent `continue` branches tied to capture failure or empty crops:

```python
if obs_frame is None or obs_frame.size == 0:
    continue
...
if img_raw is None:
    continue
...
if full_img is None:
    continue
...
if img_raw.size == 0:
    continue
```

Replace them with the strict OBS-only path:

```python
obs_frame = capture.grab()
h_scr, w_scr = obs_frame.shape[:2]
center_x_scr, center_y_scr = w_scr // 2, h_scr // 2
half_size = DETECTION_SIZE // 2
x1_crop = max(0, center_x_scr - half_size)
y1_crop = max(0, center_y_scr - half_size)
x2_crop = min(w_scr, center_x_scr + half_size)
y2_crop = min(h_scr, center_y_scr + half_size)
img_raw = obs_frame[y1_crop:y2_crop, x1_crop:x2_crop]

if img_raw.size == 0:
    raise RuntimeError("OBS Virtual Camera crop is empty")
```

- [ ] **Step 4: Re-run the strict-failure regression**

Run:

```bash
pytest test_main_aim_runtime.py::test_main_raises_immediately_when_obs_frame_read_fails -v
```

Expected: `PASSED`.

- [ ] **Step 5: Run a syntax checkpoint**

Run:

```bash
python -m py_compile main_aim.py test_main_aim_runtime.py
```

Expected: command exits with no output.

### Task 6: Run the minimum verification set

**Files:**
- No file changes

- [ ] **Step 1: Run the full automated regression set**

Run:

```bash
pytest test_main_aim_runtime.py -v
```

Expected: every test in `test_main_aim_runtime.py` is `PASSED`.

- [ ] **Step 2: Run a direct module syntax check**

Run:

```bash
python -m py_compile main_aim.py
```

Expected: command exits with no output.

- [ ] **Step 3: Run the Windows + OBS manual smoke check**

Run:

```bash
python main_aim.py --obs-camera-index=0
```

Expected manual results:
- The process starts only if OBS Virtual Camera is available.
- The `Radar` window appears.
- Left-click inside `Radar` moves the reference marker to the clicked point.
- Pressing `r` resets the reference marker to the geometric center.
- FOV circle, target selection, and yellow offset line all move relative to the calibrated point.
- Stopping OBS Virtual Camera while the program runs causes an immediate error and exit.

- [ ] **Step 4: If the manual smoke check fails, fix only the blocking bug before moving to docs**

Use this rule: if the failure is in startup, capture, calibration click, `r` reset, or strict exit on dropped frames, fix that bug before touching `README.md`.

### Task 7: Update the README to match the OBS-only calibrated-center flow

**Files:**
- Modify: `README.md:1-319`

- [ ] **Step 1: Replace the old OBS input section with OBS Virtual Camera-only instructions**

Replace the current “使用 OBS 作为画面输入” and “OBS 捕获模式” wording with this exact markdown:

```markdown
### 使用 OBS Virtual Camera 作为画面输入

`main_aim.py` 仅支持通过 OBS Virtual Camera 获取输入画面。

Windows 命令行示例：
```bat
set OBS_CAMERA_INDEX=0
python main_aim.py
```

说明：
- 程序直接读取 OBS Virtual Camera 输出画面。
- 如果 Virtual Camera 初始化失败，程序会立即报错并退出。
- 如果运行中出现读帧失败，程序会立即报错并退出。
- 不再回退到 DXCam、MSS 或 obs-websocket。
- WSL 中只能做静态/语法验证，实际 OBS 采集与鼠标联动仍需在 Windows + OBS 环境中验证。
```

- [ ] **Step 2: Update the key table and calibration instructions**

Replace the current key table with this version:

```markdown
| 按键 | 功能 |
|------|------|
| `]` | 开启/暂停自动瞄准 |
| `[` | 切换锁定目标（头/身体） |
| `\` | 切换识别窗口显示 |
| `r` | 重置参考中心到画面几何中心 |
| `q` | 退出程序 |
```

Then add this calibration section below it:

```markdown
## 参考中心校准

启动后可在 `Radar` 预览窗口中通过鼠标左键点击设置参考中心。

交互方式：
- 左键点击：设置当前参考中心
- `r`：重置参考中心，恢复为画面几何中心
- `q`：退出程序

说明：
- 目标选择以“距离当前参考中心最近”为准。
- FOV 圆心、偏移计算和鼠标移动参考点都基于当前参考中心。
- 未完成校准时，默认使用画面几何中心。
```

- [ ] **Step 3: Remove stale fallback wording everywhere else in the README**

Delete or rewrite any remaining mentions of these strings so they no longer describe current behavior:

```text
DXCam
MSS
obs-websocket
回退
Capture backend
```

- [ ] **Step 4: Read the updated README and verify the wording matches the code**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
for needle in ['DXCam', 'obs-websocket', '回退到', 'CAPTURE_BACKEND']:
    print(needle, needle in text)
PY
```

Expected:
- `DXCam False`
- `obs-websocket False`
- `回退到 False`
- `CAPTURE_BACKEND False`

---

## Follow-up backlog (not part of this implementation pass)

- Persist the calibrated center across launches.
- Support multi-click averaging before locking the center.
- Add fine adjustment hotkeys for pixel-level calibration.
- Add optional debug text for current center / offset / active target.
- Revisit runtime stability once the OBS-only path is proven in actual use.

---

## Self-review checklist

- Spec coverage:
  - OBS Virtual Camera-only startup: covered by Task 1.
  - Unified calibrated center for target selection / FOV / offset / movement: covered by Tasks 2–4.
  - Strict startup and runtime frame-failure policy: covered by Task 5.
  - Minimal runnable verification: covered by Task 6.
  - README sync: covered by Task 7.
- Placeholder scan: no `TODO`, `TBD`, or “implement later” placeholders remain in the tasks.
- Type consistency:
  - `build_runtime_config()` returns only `obs_camera_index`.
  - `ScreenCapture` constructor contract stays `target_fps, obs_camera_index`.
  - `select_closest_target()` returns a dict with `distance`, `dx`, `dy`, `box`, `cls`, `conf`.

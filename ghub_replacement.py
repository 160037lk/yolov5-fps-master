import ctypes
import time

# Windows API 结构体定义 (看不懂没关系，这是系统底层的“咒语”)
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort), ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

# 模拟罗技驱动的类结构，为了兼容你的主程序
class Logitech:
    class mouse:
        @staticmethod
        def move(x, y):
            # 0x0001 = MOUSEEVENTF_MOVE (相对移动)
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            # 这里的 int(x) 和 int(y) 确保传入整数
            ii_.mi = MouseInput(int(x), int(y), 0, 0x0001, 0, ctypes.pointer(extra))
            command = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

# 设置为 True，骗过主程序的检查
ok = True
import ctypes
import os
import sys


class Logitech:
    def __init__(self):
        self.dll = None
        self.ok = False

        # 获取当前目录下的 dll 路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dll_path = os.path.join(current_dir, 'logitech.driver.dll')

        if not os.path.exists(dll_path):
            print(f"[错误] 找不到文件: {dll_path}")
            return


        # 1. 保存当前的打印通道
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # 2. 打开一个“黑洞”文件 (空设备)
        devnull = open(os.devnull, 'w')

        try:
            # 重定向stdout和stderr
            sys.stdout = devnull
            sys.stderr = devnull

            # 加载 DLL
            self.dll = ctypes.CDLL(dll_path)

            # 尝试初始化
            try:
                self.dll.device_open()
            except:
                pass

        except Exception as e:
            # 如果出错，先恢复打印，再报错
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"[异常] 驱动加载失败: {e}")
            return

        finally:
            # 恢复打印通道
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
            self.ok = True

    def move(self, x, y):
        """ 兼容性移动指令 """
        if self.ok and self.dll:
            try:
                self.dll.moveR(int(x), int(y))
            except:
                try:
                    self.dll.mouse_event(1, int(x), int(y), 0, 0)
                except:
                    pass
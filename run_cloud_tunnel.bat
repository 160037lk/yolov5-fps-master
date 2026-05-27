@echo off
REM ============================================================
REM  云端集成推理客户端 — SSH隧道模式
REM  使用方法: 双击运行此 bat 文件
REM  前提: 确保已安装 Python 依赖 (dxcam, opencv-python, numpy)
REM ============================================================

echo [Starting] SSH Tunnel + Cloud Client
echo.

REM 启动 SSH 隧道 (后台)
echo Starting SSH tunnel to cloud server...
start /B ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "%USERPROFILE%\.ssh\cloud_rsa" -N -L 19999:127.0.0.1:9999 root@8.160.149.149 -p 53414
timeout /t 3 /nobreak >nul
echo SSH tunnel started on localhost:19999
echo.

REM 运行客户端 (通过隧道连接)
python main_aim_cloud.py --host 127.0.0.1 --port 19999 --target head

REM 清理隧道
taskkill /F /IM ssh.exe 2>nul
echo.
echo [Done] Client stopped.
pause

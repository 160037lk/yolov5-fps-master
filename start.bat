@echo off
echo ============================================================
echo   Cloud GPU Auto-Aim Client
echo ============================================================
echo.

echo [1/2] Starting SSH tunnel...
start /B ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "%USERPROFILE%\.ssh\cloud_rsa" -N -L 29999:127.0.0.1:9999 root@8.160.149.149 -p 53414
timeout /t 3 /nobreak >nul
echo        SSH tunnel: localhost:29999 -^> cloud:9999

echo [2/2] Starting client...
echo        Mouse: Interception (kernel-level)
echo        Target: HEAD
echo.
echo ============================================================
echo   Controls:
echo   ]  Toggle Aim    [  Toggle Head/Body
echo   \  Toggle Radar  Q  Quit
echo   Hold LeftAlt to aim
echo ============================================================
echo.

python main_aim_cloud.py --host 127.0.0.1 --port 29999 --target head --mouse-method interception

taskkill /F /IM ssh.exe 2>nul
echo.
echo [Done]
pause

@echo off
setlocal

REM 检查是否已经在虚拟环境中
if defined VIRTUAL_ENV (
    echo Already in virtual environment
    goto :install
)

REM 检查.venv目录是否存在
if exist .venv (
    echo Virtual environment exists
    call :activate
) else (
    echo Creating new virtual environment...
    python -m venv .venv --clear
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
    call :activate
)

:install
echo Installing/Updating packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo All done! Environment is ready.
pause
exit /b 0

:activate
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)
goto :eof
@echo off
pause
setlocal

REM Set name for virtual environment folder
set VENV_DIR=.venv

REM Check if venv exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Activate the virtual environment
call %VENV_DIR%\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Run the Streamlit app
echo Launching the app...
streamlit run semantic_tagging.py

pause

@echo off

:: Activate virtual environment if it exists
if exist venv (
    call venv\Scripts\activate
)

:: Run the application
python app.py

pause 
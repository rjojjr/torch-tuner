@echo off
call "%UserProfile%\AppData\Local\torch-tuner\.venv\Scripts\activate.bat"
set args=%*
python3.11 "%UserProfile%\AppData\Local\torch-tuner\src\main\main.py" %args%
call deactivate
@echo off
echo "Uninstalling Torch Tuner CLI"

del /s /q "%UserProfile%\AppData\Local\torch-tuner"
rmdir /s /q "%UserProfile%\AppData\Local\torch-tuner"
del /q C:\Windows\System32\torch-tuner
echo "Uninstalled Torch Tuner CLI"
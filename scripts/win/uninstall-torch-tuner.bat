echo on
echo "Uninstalling Torch Tuner CLI"

del /s /q "%UserProfile%\.local\torch-tuner"
rmdir /s /q "%UserProfile%\.local\torch-tuner"
del /q C:\Windows\System32\torch-tuner
echo "Uninstalled Torch Tuner CLI"
@echo off
echo "Installing Torch Tuner CLI"

pushd .

echo "Preparing installation directory"
if exist "%UserProfile%\AppData\Local\torch-tuner" @RD /S /Q "%UserProfile%\AppData\Local\torch-tuner"
cd "%UserProfile%\AppData\Local"
echo "Getting latest CLI from github"
git clone https://github.com/rjojjr/torch-tuner.git

cd torch-tuner

echo "Installing python dependencies"

python3.11 -m venv .\.venv && call .\.venv\Scripts\activate.bat
python3.11 -m pip install -I -r requirements.in --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us
call deactivate

echo "Finalizing install"

echo f | xcopy /f /y scripts\win\torch-tuner.bat C:\Windows\System32\torch-tuner.bat

icacls "%UserProfile%\AppData\Local\torch-tuner" /grant Users:F
attrib -s C:\Windows\System32\torch-tuner.bat

popd

echo "Torch Tuner CLI installed successfully!"
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."


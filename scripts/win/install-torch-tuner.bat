echo on
echo "Installing Torch Tuner CLI"
echo ""
CD "%UserProfile%\.local"


if exist "%UserProfile%\.local\torch-tuner" @RD /S /Q "%UserProfile%\.local\torch-tuner"

git clone https://github.com/rjojjr/torch-tuner.git

cd torch-tuner

git checkout create-windows-os-installer-script

python -m venv .\.venv && .\.venv\Scripts\activate.bat
pip install -I -r requirements.txt
deactivate

xcopy scripts\win\torch-tuner C:\Windows\System32\torch-tuner

echo ""
echo "Torch Tuner CLI installed successfully!"
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."


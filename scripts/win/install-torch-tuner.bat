echo on
echo "Installing Torch Tuner CLI"
echo ""

echo "Preparing installation directory"
echo ""
if exist "%UserProfile%\.local\torch-tuner" @RD /S /Q "%UserProfile%\.local\torch-tuner"
CD "%UserProfile%\.local"
echo "Getting latest CLI from github"
echo ""
git clone https://github.com/rjojjr/torch-tuner.git

cd torch-tuner

git checkout create-windows-os-installer-script
echo ""
echo "Installing python dependencies"
echo ""
python -m venv .\.venv && .\.venv\Scripts\activate.bat
pip install -I -r requirements.txt
deactivate

echo ""
echo "Finalizing install"
echo ""

xcopy scripts\win\torch-tuner C:\Windows\System32\torch-tuner

icacls "%UserProfile%\.local\torch-tuner" /grant Users:F
attrib -s C:\Windows\System32\torch-tuner

echo ""
echo "Torch Tuner CLI installed successfully!"
echo "You can now access the Torch Tuner CLI with the 'torch-tuner' command."


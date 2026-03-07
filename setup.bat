@echo off
:: Set up Python 3.13 environment as Parallax was developed under it.
:: You may need to edit this file to set the path to your global
:: Python 3.13 location prior to execution of this batch script.
:: Also, make sure you have the CUDA Toolkit 13.0 installed as your
:: primary toolkit. Flash Attention 2 was compiled under these specific
:: versions and using differing versions will cause a mismatch and errors.

:: Create virtual environment
echo "Creating virtual environment..."
C:\Python313\python -m venv .venv
call .venv\Scripts\activate.bat

:: Make directories for project
echo "Creating project directories..."
mkdir .vscode data logs checkpoints

:: Create 'placeholder' files
echo "Creating placeholder files..."
echo;>data/.data-files-placed-here
echo;>logs/.log-files-saved-here
echo;>checkpoints/.checkpoint-files-saved-here

:: Create VSCode settings files (Windows)
echo Creating VSCode Windows settings...
(
echo {
echo     // Python-specific Formatting ^& Analysis
echo     "[python]": {
echo         "editor.defaultFormatter": "ms-python.black-formatter",
echo         "editor.formatOnSave": true,
echo         "editor.codeActionsOnSave": {
echo             "source.organizeImports": "always"
echo         }
echo     },
echo     "python.analysis.typeCheckingMode": "basic",
echo.
echo     // Editor Polish
echo     "editor.bracketPairColorization.enabled": true,
echo     "editor.guides.bracketPairs": "active",
echo     "editor.smoothScrolling": true,
echo     "editor.minimap.enabled": false,
echo.
echo     // File Hygiene
echo     "files.insertFinalNewline": true,
echo     "files.trimTrailingWhitespace": true,
echo     "files.autoSave": "onFocusChange"
echo }
) > ./.vscode/settings.json

:: Create shortcuts to Activate & Deactivate environment
echo "Creating activate/deactivate shortcut scripts..."
echo @echo off > a.bat
echo call .venv\Scripts\activate.bat >> a.bat
echo @echo off > d.bat
echo call .venv\Scripts\deactivate.bat >> d.bat

:: Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
python.exe -m pip install --upgrade pip setuptools

:: Create requirements.txt file
echo Writing requirements.txt...
(
    echo huggingface_hub[hf-xet]
	echo accelerate==1.12.0
	echo bitsandbytes==0.49.2
	echo blobfile==3.2.0
	echo cupy-cuda13x==14.0.1
	echo dataclasses==0.6
	echo datasets==4.6.1
	echo diffusers==0.36.0
	echo einops==0.8.2
	echo librosa==0.11.0
	echo matplotlib==3.10.8
	echo ninja==1.13.0
	echo nvitop==1.6.2
	echo peft==0.18.1
	echo tensorboard==2.20.0
	echo tiktoken==0.12.0
	echo torchao==0.16.0
	echo torchtext==0.6.0
	echo transformers==5.2.0
	echo triton-windows==3.6.0.post25
	echo trl==0.29.0
) > requirements.txt

:: Install PyTorch with CUDA 13.0 support
echo "Installing PyTorch with CUDA 13.0... (this may take a while)"
pip install xformers==0.0.34 torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 torchmetrics==1.0.3 -i https://download.pytorch.org/whl/cu130

:: Install other dependencies
echo "Installing other dependencies from requirements.txt... (this may take a while longer!)"
pip install -r requirements.txt

:: Install Flash Attention 2
echo Installing Flash Attention 2
pip install whl\flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp313-cp313-win_amd64.whl

:: Install Parallax model package as an editable install so that
:: "from model.parallax import ..." works from any script location
:: without requiring sys.path manipulation.
echo Installing Parallax as editable package...
pip install -e .

echo Virtual environment created and configured for Parallax as it was when developed under Windows 11.
echo If using Microsoft Visual Studio Code, you can start by entering the below on the command line:
echo code .

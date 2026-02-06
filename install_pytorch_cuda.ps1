# Install PyTorch with CUDA 12.1 support
# Run this script to install the correct PyTorch version with CUDA support

Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green

# Install PyTorch packages with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nPyTorch installation complete!" -ForegroundColor Green
Write-Host "Verifying installation..." -ForegroundColor Yellow

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"




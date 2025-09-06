# Quick Start

> **Recommended Python:** 3.10 or 3.11 (3.12 works for most stacks but may have fewer prebuilt wheels).

## 1) Create & activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools (recommended):

```bash
python -m pip install -U pip setuptools wheel
```

## 2) Install the package (editable mode)
Instead of juggling relative imports, this repo uses a `pyproject.toml`.
Install everything into your venv:

```bash
pip install -e
```
This registers `src/` as a proper package (`trend-prediction`), so you can import modules anywhere without touching `PYTHONPATH`.

## 3) Install ONE platform-specific Torch stack

Choose exactly one of the following:

### a) macOS (Apple Silicon or Intel) – CPU/MPS (no CUDA) OR Windows / Linux – CPU-only

```bash
pip install -r requirements-cpu.txt
```

### b) Windows / Linux – NVIDIA GPU (CUDA 11.8)

```bash
# Use the PyTorch CUDA 11.8 wheel index for Torch/TorchVision/Torchaudio
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-cuda118.txt

# If torch-geometric (PyG) extras are installed separately and need specific wheels:
# pip install -f https://data.pyg.org/whl/torch-2.7.1+cu118.html torch-geometric
```

> If you see **“No matching distribution found for torch==…+cu118”** on a **Mac**, you’ve selected the CUDA file by mistake. Use `requirements-mac.txt` instead.

## 4) Install Playwright browsers (once)

```bash
playwright install
```

This is required for Playwright-based scrapers (i.e. TikTok).

## 5) (Optional) Verify Torch backend

```bash
python - <<'PY'
import torch, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available (macOS):", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
PY
```

## 6) Run the app

```bash
python -m service.main
```
(You can also run it as python src/service/main.py, but the -m form ensures package-relative imports work consistently.)

## 7) Run tests
Tests live in the `tests/` folder at repo root. After installation, just run:
```bash
pytest
```
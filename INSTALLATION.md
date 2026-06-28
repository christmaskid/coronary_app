# Coronary Stenosis Viewer - Installation Guide

## Overview

The Coronary Stenosis Viewer is a Streamlit-based interactive application for analyzing coronary artery stenosis from CT images. This guide walks through environment setup and installation.

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.9 or higher (3.12 recommended)
- **RAM**: 8 GB minimum (16 GB recommended for large datasets)
- **GPU**: Optional (CUDA 11.8+ for acceleration)

## Quick Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate coronary_env
```

### 2. Run the Application

```bash
streamlit run coronary_app/coronary_stenosis_viewer_app.py
```

The app will be available at `http://localhost:8501` or other port that shows up.

---

## Detailed Installation Steps

### Option A: Using Conda (Recommended)

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd coronary_app
```

#### Step 2: Create Environment from YAML
```bash
conda env create -f environment.yml -n coronary_env
conda activate coronary_env
```

#### Step 3: Verify Installation
```bash
python -c "import streamlit; import nibabel; import scipy; print('✓ All core dependencies installed')"
```

#### Step 4: Configure Streamlit (Optional)
Create `~/.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "none"
runOnSave = false
headless = true

[logger]
level = "info"
```

#### Step 5: Launch Application
```bash
streamlit run coronary_app/coronary_stenosis_viewer_app.py
```

---

### Option B: Manual Package Installation

If you prefer to install without conda:

```bash
# Create virtual environment
python3.12 -m venv coronary_env
source coronary_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run coronary_app/coronary_stenosis_viewer_app.py
```

---

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21 | Numerical operations |
| scipy | >=1.9 | Scientific computing (interpolation, signal processing) |
| scikit-image | >=0.20 | Image morphology & skeletonization |
| nibabel | >=5.0 | NIfTI file I/O |
| networkx | >=3.0 | Graph operations for vessel network |
| plotly | >=5.0 | 3D visualization |
| streamlit | >=1.28 | Web UI framework |
| pandas | >=1.5 | Data handling (metadata) |
| Cython | >=0.29 | Extension compilation |

---

## Troubleshooting

### Issue: `inotify instance limit reached`
**Solution**: Disable Streamlit file watching
```bash
streamlit run coronary_stenosis_viewer_app.py --server.fileWatcherType none
```

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
**Solution**: Reinstall streamlit
```bash
pip install --upgrade streamlit
```
---

## Running on Remote Server (SSH)

### Via SSH with X11 Forwarding:
```bash
ssh -X user@server.com
conda activate coronary_env
streamlit run coronary_stenosis_viewer_app.py
```

### Via SSH without Display (Headless):
```bash
# On server:
streamlit run coronary_stenosis_viewer_app.py --server.headless true --server.port 8501

# On local machine:
ssh -L 8501:localhost:8501 user@server.com
# Then visit http://localhost:8501
```

---

## Testing the Installation

```bash
# Check all imports work
python -c "
import streamlit as st
import nibabel as nib
import numpy as np
import scipy
import sklearn
import plotly.graph_objects as go
import networkx as nx
print('✓ All imports successful')
"

# Verify Streamlit configuration
streamlit config show

# Test file watching is disabled
cat ~/.streamlit/config.toml
```

---

## Performance Notes

- **Mask Preprocessing**: ~2-5 seconds (size-dependent)
- **Path Extraction**: ~5-15 seconds
- **Straightening (multi-angle)**: ~10-30 seconds
- **Total Analysis**: ~20-50 seconds (typical)

For faster performance on large datasets, consider:
- Reducing CT volume size (crop ROI)
- Decreasing `output_plane_size` in settings
- Reducing number of angles for stenosis measurement

---

## License & Attribution
With aid of Github Copilot.

Last Updated: 2026-06-26

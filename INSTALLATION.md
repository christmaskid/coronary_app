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

### 2. Install the Package

```bash
cd coronary_postprocessing
python setup.py build_ext --inplace
```

### 3. Run the Application

```bash
streamlit run coronary_stenosis_viewer_app.py
```

The app will be available at `http://localhost:8501`

---

## Detailed Installation Steps

### Option A: Using Conda (Recommended)

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd coronary_postprocessing
```

#### Step 2: Create Environment from YAML
```bash
conda env create -f environment.yml -n coronary_env
conda activate coronary_env
```

#### Step 3: Compile Cython Extensions
```bash
python setup.py build_ext --inplace
```

#### Step 4: Verify Installation
```bash
python -c "import streamlit; import nibabel; import scipy; print('✓ All core dependencies installed')"
```

#### Step 5: Configure Streamlit (Optional)
Create `~/.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "none"
runOnSave = false
headless = true

[logger]
level = "info"
```

#### Step 6: Launch Application
```bash
streamlit run coronary_stenosis_viewer_app.py
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

# Compile extensions
python setup.py build_ext --inplace

# Run application
streamlit run coronary_stenosis_viewer_app.py
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

### Issue: Cython compilation fails
**Solution**: Install build dependencies
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install
```

### Issue: Results disappear when moving sliders
**Solution**: This is already fixed in the latest version (uses `st.session_state` persistence)

---

## Recent Changes (Coronary Viewer Enhancements)

### Features Added:
1. **Remote File Browsing**: Browse NIfTI files directly from SSH server
2. **Result Persistence**: Analysis results stay on screen when adjusting parameters
3. **Save/Load Bundles**: Auto-save analysis results as compressed `.npz` + metadata JSON
4. **Proportional CT Display**: Fixed aspect ratio (1:1) for CT slice visualization
5. **Consensus Stenosis Table**: Shows detected stenosis regions with vote ratios
6. **Result Download**: Download saved analysis bundles directly from UI
7. **Save Log**: Track recently saved results in sidebar
8. **Progress Animation**: Multi-stage status display during analysis
9. **Multi-mode Input**: Upload local files, browse server, or reopen saved results
10. **Settings Persistence**: All analysis parameters are saved with results

### UI Improvements:
- Mode-specific sidebar that changes based on "Upload", "Browse", or "Open" mode
- Consensus stenosis analysis table in results panel
- Download button for saved bundles
- Live save log showing where results were stored
- Window level/width controls work without recomputation

---

## File Structure

```
coronary_postprocessing/
├── coronary_stenosis_viewer_app.py    # Main Streamlit application
├── setup.py                            # Cython extension build config
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # Pip requirements
├── INSTALLATION.md                     # This file
├── viewer_results/                     # Saved analysis bundles (auto-created)
├── masks/                              # Test data directory
└── [notebooks]                         # Legacy Jupyter notebooks
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

## Support & Documentation

For detailed algorithm information, see included Jupyter notebooks:
- `Straightened-GT_manual-refactored.ipynb` - Core straightening logic
- `stenosis_detection_pipeline_IPA.ipynb` - Multi-angle stenosis detection

---

## License & Attribution

[Add your project license and attribution here]

Last Updated: 2026-06-26

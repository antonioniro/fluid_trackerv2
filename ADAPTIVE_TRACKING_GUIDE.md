# Adaptive Fluid Surface Tracking Guide

## 🌊 Overview
This enhanced fluid tracking system automatically adapts to different videos with varying lighting conditions, fluid colors, and container shapes. No more manual parameter tuning for each video!

## 🚀 Key Improvements

### 1. **Interactive Color Selection**
- Click on the fluid to automatically detect its color
- Handles different lighting conditions
- Samples multiple pixels for better accuracy
- Automatic HSV range calculation

### 2. **Color Profile Management**
- Automatically saves color profiles for each video
- Loads existing profiles on subsequent runs
- No need to reselect colors every time
- Profiles stored in `color_profiles/` directory

### 3. **Adaptive Lighting Adjustment**
- Automatically adjusts parameters based on scene brightness
- Handles dark, bright, and low-saturation conditions
- Updates color ranges in real-time during analysis

### 4. **Enhanced Detection**
- Better edge filtering to ignore container walls
- Improved surface interpolation
- Handles hue wrap-around (e.g., red colors)
- Multiple color range support

### 5. **Flexible Analysis Modes**
- **Full Frame Mode**: Analyzes the entire video frame automatically
  - Best for: Simple setups, fluid fills most of the frame
  - Faster setup, no manual region selection needed
- **ROI Mode**: Allows manual selection of specific regions
  - Best for: Complex scenes, multiple objects, specific focus areas
  - More precise analysis, better performance

## 📋 Usage Instructions

### Quick Start
```bash
# Run the main script
python fluid_tracking.py

# Or use the demo
python demo_adaptive_tracking.py
```

### Basic Python Usage
```python
from fluid_tracking import FluidTracker

# Create tracker
tracker = FluidTracker("your_video.mp4", "output_video.mp4")

# Run analysis (will prompt for ROI and color selection)
tracker.analyze_wave_motion(debug_mode=True)

# Show results
tracker.plot_surface_analysis()
```

## 🎯 Step-by-Step Workflow

### First Time with a Video:
1. **Video Selection**: Choose your video file
2. **Mode Selection**: Choose between:
   - **Mode 1**: Full Frame Mode (use entire video frame)
   - **Mode 2**: ROI Mode (manually select region of interest)
3. **ROI Selection** (only for Mode 2): Click and drag to select the area containing the fluid
4. **Color Selection**: Click on the fluid to sample its color
5. **Analysis**: The system tracks the fluid surface automatically
6. **Results**: View real-time analysis and final plots

### Subsequent Runs:
1. **Auto-load**: Color profile is loaded automatically
2. **Mode Selection**: Choose your preferred analysis mode
3. **Analysis**: Tracking starts immediately
4. **Adjustment**: Press 'c' during analysis to reselect colors if needed

## ⌨️ Controls During Analysis

| Key | Action |
|-----|--------|
| `q` | Quit analysis |
| `p` | Pause/Resume |
| `s` | Save current frame |
| `r` | Reset tracking points |
| `d` | Toggle debug mode |
| `c` | Reselect fluid color |

## 🔧 Advanced Features

### Color Profile Format
Color profiles are stored as JSON files with:
- Target HSV and BGR colors
- Adaptive HSV ranges
- Tolerance parameters
- Hue wrap-around handling

### Lighting Adaptation
The system automatically adjusts:
- **Dark conditions**: Increases value tolerance, reduces minimum value
- **Bright conditions**: Decreases value tolerance, increases minimum value
- **Low saturation**: Increases saturation tolerance, reduces minimum saturation

### Debug Mode
Shows:
- Detection mode (Adaptive vs Default)
- Color masks and ranges
- Edge exclusion zones
- Target color information

## 📊 Output Files

### Generated Files:
- `analisi_fluido_output_[videoname].mp4` - Analyzed video with overlays
- `color_profiles/[videoname]_color_profile.json` - Color profile
- Analysis plots showing surface metrics over time

### Analysis Plots:
1. **Mean Surface Height**: Track vertical movement
2. **Height Range**: Measure surface stability
3. **Standard Deviation**: Assess surface roughness
4. **Movement Velocity**: Calculate motion speed

## 🎨 Supported Scenarios

### Works Well With:
- ✅ Different lighting conditions
- ✅ Various fluid colors
- ✅ Multiple container shapes
- ✅ Different video qualities
- ✅ Varying fluid viscosities

### Limitations:
- ❌ Extremely low contrast scenes
- ❌ Heavily occluded fluids
- ❌ Very fast motion blur
- ❌ Multiple fluid types in same scene

## 🔍 Troubleshooting

### Poor Detection Rate:
1. **Check lighting**: Ensure adequate contrast
2. **Adjust ROI**: Select a tighter region around the fluid
3. **Reselect colors**: Press 'c' during analysis
4. **Enable debug mode**: Press 'd' to see detection masks

### Color Selection Issues:
1. **Click on pure fluid**: Avoid container edges
2. **Multiple samples**: Click on different fluid areas
3. **Good lighting**: Select colors in well-lit areas
4. **Reset if needed**: Press 'r' in color selection mode

### Performance Tips:
- Use smaller ROI for faster processing
- Enable debug mode only when needed
- Save color profiles for repeated use
- Close other applications for better performance

## 📁 File Structure
```
fluid_tracker_v2/
├── fluid_tracking.py          # Main tracking system
├── demo_adaptive_tracking.py  # Interactive demo
├── color_profiles/            # Saved color profiles
├── your_videos.mp4           # Input videos
└── analisi_fluido_output_*.mp4 # Output videos
```

## 🛠️ Technical Details

### Color Detection Algorithm:
1. Convert to HSV color space
2. Create adaptive color ranges based on samples
3. Handle hue wrap-around for edge colors
4. Apply morphological operations for cleanup
5. Filter out container edges and noise

### Surface Tracking:
1. Find fluid contours in color mask
2. Extract topmost points as surface line
3. Filter outliers and interpolate gaps
4. Apply smoothing for stability
5. Track motion using optical flow

### Adaptive Parameters:
- **Hue tolerance**: ±20° around target
- **Saturation tolerance**: ±60 units
- **Value tolerance**: ±80 units (adaptive)
- **Color tolerance**: 40 units in BGR space

## 🤝 Contributing
Feel free to improve the system by:
- Adding support for new color spaces
- Implementing machine learning detection
- Optimizing performance
- Adding new analysis metrics

## 📝 License
This project is for educational and research purposes. 
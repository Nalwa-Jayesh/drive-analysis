# AthleteRise - AI-Powered Cricket Analytics

Real-time cricket cover drive analysis system using computer vision and pose estimation.

## Overview

This system processes cricket videos frame-by-frame to analyze batting technique, specifically focusing on cover drive shots. It provides:

- **Real-time pose estimation** using MediaPipe
- **Biomechanical metrics calculation** (elbow angle, spine lean, alignment, etc.)
- **Live video overlays** with technique feedback
- **Comprehensive shot evaluation** with actionable insights
- **Phase detection** (Stance → Stride → Downswing → Impact → Follow-through → Recovery)
- **Temporal analysis charts** showing technique consistency over time

## Features

### Base Requirements ✅
- [x] Full video processing with real-time flow
- [x] Per-frame pose estimation using MediaPipe
- [x] Biomechanical metrics calculation
- [x] Live overlays in output video
- [x] Final shot evaluation with scores 1-10
- [x] Graceful handling of missing detections

### Bonus Features Implemented 🎯
- [x] **Automatic Phase Segmentation** - Detects batting phases using joint angles
- [x] **Temporal Smoothness Analysis** - Generates consistency charts
- [x] **Real-Time Performance Target** - Optimized for ≥10 FPS processing
- [x] **Skill Grade Prediction** - Maps metrics to Beginner/Intermediate/Advanced
- [x] **Robustness & UX** - Comprehensive error handling and logging
- [x] **YouTube Video Download** - Direct analysis from YouTube URLs
- [x] **Comprehensive Reporting** - JSON evaluation with detailed feedback

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://www.github.com/Nalwa-Jayesh/drive-analysis.git
cd drive-analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

#### Analyze YouTube Video (Default)
```bash
python cover_drive_analysis_realtime.py
```
This will analyze the default YouTube Short: https://youtube.com/shorts/vSX3IRxGnNY

#### Analyze Custom YouTube Video
```bash
python cover_drive_analysis_realtime.py --url "https://youtube.com/shorts/YOUR_VIDEO_ID"
```

#### Analyze Local Video File
```bash
python cover_drive_analysis_realtime.py --video "path/to/your/video.mp4"
```

#### Custom Output Directory
```bash
python cover_drive_analysis_realtime.py --output "custom_output_dir"
```

## Output Files

The system generates several output files in the specified directory:

- **`annotated_video.mp4`** - Full video with pose overlays and real-time metrics
- **`evaluation.json`** - Comprehensive shot analysis with scores and feedback
- **`temporal_analysis.png`** - Charts showing technique consistency over time

## Analysis Metrics

### Core Biomechanical Metrics
- **Front Elbow Angle** - Shoulder-elbow-wrist angle (optimal: 100-140°)
- **Spine Lean** - Trunk lean from vertical (optimal: 5-25°)
- **Head-Knee Alignment** - Horizontal distance between head and front knee
- **Front Foot Direction** - Foot angle relative to batting direction

### Evaluation Categories
- **Footwork** (1-10) - Phase transitions and balance
- **Head Position** (1-10) - Steadiness and alignment
- **Swing Control** (1-10) - Elbow position and bat path
- **Balance** (1-10) - Spine angle consistency
- **Follow-through** (1-10) - Completion of shot

## Technical Architecture

### Core Components

1. **CricketPoseAnalyzer** - Main analysis class
   - MediaPipe pose estimation
   - Biomechanical calculations
   - Phase detection algorithms

2. **Video Processing Pipeline**
   - Frame-by-frame analysis
   - Real-time overlay rendering
   - Performance optimization

3. **Evaluation Engine**
   - Multi-category scoring system
   - Actionable feedback generation
   - Skill level prediction

### Performance Optimizations

- **Model Selection** - MediaPipe Pose (complexity=1) for speed/accuracy balance
- **Frame Processing** - Efficient OpenCV operations
- **Memory Management** - Circular buffers for metrics tracking
- **Target Performance** - ≥10 FPS on CPU

## Configuration

The system uses configurable thresholds for technique evaluation:

```python
config = {
    'thresholds': {
        'good_elbow_angle': (100, 140),      # degrees
        'good_spine_lean': (5, 25),          # degrees  
        'good_alignment_threshold': 50,       # pixels
        'good_foot_angle': (-30, 30)         # degrees
    }
}
```

## Sample Output

### Console Output
```
Processing video: input_video.mp4
Video specs: 1080x1920, 30.0 FPS, 180 frames
Progress: 16.7% (30/180)
Progress: 33.3% (60/180)
...
Processing complete!
Total time: 12.34s
Average FPS: 14.58

==================================================
ANALYSIS COMPLETE!
==================================================
Output video: output/annotated_video.mp4
Evaluation file: output/evaluation.json
Processing FPS: 14.58
Target FPS achieved: ✅

Overall Score: 7.2/10
Skill Level: Intermediate

Category Scores:
  Footwork: 8.1/10
  Head Position: 7.5/10
  Swing Control: 6.8/10
  Balance: 7.3/10
  Follow Through: 6.5/10
```

### Evaluation JSON Structure
```json
{
  "timestamp": "2024-01-15 10:30:45",
  "total_frames_analyzed": 180,
  "overall_score": 7.2,
  "skill_level": "Intermediate",
  "scores": {
    "footwork": 8.1,
    "head_position": 7.5,
    "swing_control": 6.8,
    "balance": 7.3,
    "follow_through": 6.5
  },
  "feedback": {
    "footwork": [
      "Good phase progression through the shot",
      "Good foot positioning"
    ],
    "head_position": [
      "Good head position maintained",
      "Watch the ball closely through contact"
    ]
  },
  "phase_distribution": {
    "Stance": 25.0,
    "Stride": 15.0,
    "Downswing": 20.0,
    "Impact": 10.0,
    "Follow-through": 20.0,
    "Recovery": 10.0
  },
  "summary_stats": {
    "avg_elbow_angle": 118.5,
    "avg_spine_lean": 15.2,
    "avg_alignment_distance": 35.8,
    "consistency_score": 8.2
  }
}
```

## Assumptions & Limitations

### Assumptions
- **Batsman Orientation** - Assumes right-handed batsman (left side as front)
- **Camera Angle** - Works best with side-on camera angles
- **Video Quality** - Requires clear view of batsman's full body
- **Lighting** - Consistent lighting for reliable pose detection

### Current Limitations
- **Bat Tracking** - Not implemented in base version
- **Ball Tracking** - Contact moment detection is heuristic-based
- **Multi-person** - Designed for single batsman analysis
- **Court/Ground** - No pitch/crease line detection for absolute measurements

### Future Enhancements
- Advanced bat detection using YOLO/custom models
- Ball trajectory analysis and contact point detection
- Multi-camera angle fusion
- Comparative analysis against professional templates
- Real-time streaming analysis

## Development

### Project Structure
```
├── cover_drive_analysis_realtime.py    # Main analysis script
├── requirements.txt                     # Dependencies
├── README.md                           # Documentation
├── output/                             # Generated outputs
│   ├── annotated_video.mp4
│   ├── evaluation.json
    └── temporal_analysis.png
```


## Troubleshooting

### Common Issues

1. **Video Download Fails**
   - Check internet connection
   - Verify YouTube URL is accessible
   - Try different video quality settings

2. **Pose Detection Poor**
   - Ensure good lighting in video
   - Check if full body is visible
   - Try different MediaPipe model complexity

3. **Low Processing FPS**
   - Reduce video resolution
   - Lower MediaPipe model complexity
   - Close other applications

4. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### System Requirements
- **Python** 3.8+
- **RAM** 4GB+ recommended
- **CPU** Multi-core recommended for real-time processing
- **Storage** 500MB+ for video processing

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for pose estimation models
- OpenCV community for computer vision tools
- Cricket coaching community for technique insights

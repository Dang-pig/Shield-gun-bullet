# Shield, Gun, Bullet

A real-time hand gesture recognition game for two players using computer vision and machine learning. Show hand gestures to battle with shields, guns, and bullets in this strategic gesture-based combat game!

![Game Demo](demo.gif)

*Demo GIF: Add your game demonstration video as `demo.gif` in the root directory*

## Game Overview

**Shield, Gun, Bullet** is a competitive hand gesture recognition game where two players face off using:
- **Shield**: Block incoming shots
- **Gun**: Shoot opponents (requires bullets)
- **Bullet**: Reload ammunition for future shots

The game combines rock-paper-scissors strategy with real-time computer vision to create an engaging, interactive experience.

## Project Structure

### Root Files
- **`__main__.py`** - Main game application with complete UI and game logic
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation file

### Model Directory (`model/`)
- **`classifier.model`** - Trained XGBoost model for gesture classification
- **`hand_landmarker.task`** - MediaPipe hand detection model

### Training Directory (`Train/`)
- **`prepareData.py`** - Data collection script for gesture training
- **`trainml.ipynb`** - Jupyter notebook for model training and evaluation
- **`test.py`** - Testing script for gesture recognition
- **`data/`** - Directory containing training data files

## Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam/camera
- Sufficient lighting for hand detection

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Models
Place the following files in the `model/` directory:
- `classifier.model` - Trained gesture classification model
- `hand_landmarker.task` - MediaPipe hand landmark model

## How the Game Works

### Core Mechanics

#### Gesture Recognition
- **Shield**: Open palm facing camera
- **Gun**: Closed fist
- **Bullet**: Pointing finger (peace sign)
- **Unknown**: Unclear or unrecognized gestures

#### Round Structure
1. **Preparation Phase**: "Shield... Gun... Bullet..." countdown
2. **Camera Phase**: Players show their gestures simultaneously
3. **Resolution Phase**: Gestures are analyzed and effects applied

#### Strategic Elements

**Bullet Management:**
- Show **Bullet** gesture to gain ammunition (+1 bullet)
- Show **Gun** gesture to shoot (requires bullets, consumes 1)
- Show **Shield** gesture to block shots

**Warning System:**
- Shooting without bullets = Bullet warning
- Using Shield more than 5 times in a row = Shield warning
- 3 warnings of any type = Loss

**Critical Hit Mechanic:**
- If Player A shoots while Player B is holding Bullet, Player B loses immediately!

### Special Rules

#### Unknown Gesture Penalty
- If **ANY** player shows an unrecognized gesture, **BOTH** players receive:
  - Zero bullet additions/removals
  - Zero warning additions
- This prevents exploitation and ensures fair play

#### Simultaneous Gesture Requirement
- Both players must show valid gestures for bullet mechanics to apply
- Encourages active participation and clear gesture recognition

## Running the Game

### Quick Start
```bash
python __main__.py
```

### Controls
- **Mouse**: Click menu buttons
- **Camera**: Show hand gestures in your screen half (left/right split)
- **Q**: Quit game
- **F**: Toggle fullscreen
- **P**: Pause game
- **H**: Return to home screen

### Game Flow
1. **Home Screen**: Click "PLAY" to start
2. **Instructions**: Read rules or click "PLAY" again
3. **Game Start**: Follow countdown phases
4. **Active Play**: Show gestures during camera phase
5. **Results**: View round outcomes and repeat
6. **Game End**: When someone loses, view results and return home

## Data Preparation & Training

### 1. Collect Training Data
```bash
cd Train
python prepareData.py
```

This script will:
- Capture hand gestures via webcam
- Save landmark data for each gesture type
- Create training files in the `data/` directory

### 2. Train the Model
Open `trainml.ipynb` in Jupyter and run all cells:
```bash
jupyter notebook trainml.ipynb
```

The notebook will:
- Load collected training data
- Train an XGBoost classifier
- Evaluate model performance
- Save the trained model to `../model/classifier.model`

### 3. Test Gesture Recognition
```bash
cd Train
python test.py
```

This will test real-time gesture recognition using your trained model.

## Technical Details

### Hand Detection Pipeline
1. **MediaPipe Hand Landmarker**: Detects 21 hand landmarks
2. **Feature Extraction**: Normalizes landmark coordinates
3. **XGBoost Classification**: Predicts gesture classes
4. **Real-time Processing**: 30+ FPS on modern hardware

### Game Architecture
- **Object-oriented UI system** with hierarchical rendering
- **State machine** for game flow management
- **Split-screen processing** for two-player support
- **Conditional logic** for fair gesture-based mechanics

### Model Performance
- **Gesture Classes**: Shield, Gun, Bullet, Unknown
- **Input Features**: 63 normalized landmark coordinates (21 points × 3 dimensions)
- **Accuracy**: 90%+ on clean gesture data
- **Real-time**: Optimized for live gameplay

## Game Features

### Visual Elements
- **Split-screen display** with player zones
- **Real-time hand landmark visualization** (green dots)
- **Dynamic UI updates** for bullets and warnings
- **Animated countdown** phases
- **Result screens** with clear win/loss feedback

### Audio-Visual Feedback
- **Text-based feedback** for all actions
- **Warning indicators** for rule violations
- **Last round results** displayed on home screen
- **Smooth transitions** between game states

## Troubleshooting

### Common Issues

**"Cannot open camera"**
- Ensure webcam is connected and not used by other applications
- Check camera permissions for Python applications

**Poor gesture recognition**
- Ensure good lighting on hands
- Hold gestures steady during camera phase
- Retrain model with your specific hand/gesture data

**Low performance**
- Close other applications using camera
- Ensure sufficient RAM (4GB+ recommended)
- Update graphics drivers

### Model Training Tips
- Collect 50-100 samples per gesture class
- Use consistent lighting and hand positioning
- Include various hand orientations
- Test with multiple users for better generalization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

### Areas for Improvement
- Mobile/web deployment
- Additional gesture types
- Voice feedback
- Multiplayer networking
- Advanced AI opponents

## License

This project is open source. Feel free to use, modify, and distribute.

## Acknowledgments

- **MediaPipe** for hand detection
- **XGBoost** for gesture classification
- **OpenCV** for computer vision
- **Scikit-learn** for preprocessing

---

**Ready to battle? Show your gestures and may the best strategist win!**

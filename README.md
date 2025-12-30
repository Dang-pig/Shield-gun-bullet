Shield, Gun, Bullet
================================================

Overview
--------
Shield, Gun, Bullet is a two-player, gesture-controlled combat game. 
The system uses MediaPipe for hand landmark detection and an XGBoost 
classifier to interpret player actions in real-time.

Hand Gestures
-------------
1. Shield: Open palm with fingers pressed together. The palm or the 
   back of the hand can face the camera; thumb position is optional.
   Function: Blocks incoming shots.

2. Gun: Index and middle fingers extended and pressed together; 
   pinky and ring fingers closed. Thumb can point up or be tucked.
   Function: Fires a shot (Consumes 1 ammunition).

3. Bullet: A closed fist (thumb can point up or be tucked). 
   Function: Reloads (Adds 1 ammunition).

Game Structure
--------------
Matches are governed by a state machine to ensure synchronized turns:

1. Countdown: A "READY - SET - GO" sequence (1s per state).
2. Chant Phase: Visual cues for "Shield... Gun... Bullet..." at 
   0.5s intervals to establish timing.
3. Reveal Phase: A 1s window where the camera captures hand 
   landmarks for both players.
4. Resolution: The system compares gestures and updates the HUD.
5. Result Display: Shows the round outcome for 1.2s before 
   resetting or declaring a winner.

Rules and Penalties
-------------------
* Critical Hit: Shooting (Gun) while an opponent reloads (Bullet) 
  results in an instant win.
* Ammunition Warning: Triggered by attempting to shoot with 0 bullets.
* Shield Fatigue: Triggered by using the Shield gesture more than 
  5 times consecutively.
* Game Over: Accumulating 3 warnings of any type results in a loss.
* Fair Play Guard: If an "Unknown" gesture is detected, the round 
  is voided (no ammunition or warnings are updated).

Data Preparation
----------------
To collect training data for the gesture classifier, use 'prepareData.py':

1. Set the SYMBOL variable in the script to the gesture you wish 
   to record (e.g., "shield", "gun", "bullet", or "unknown").
2. Run the script: python prepareData.py
3. Position your hand in the camera view. When the hand is 
   detected (blue landmarks appear), press 'C' to capture a frame.
4. Aim for 100-200 captures per gesture, varying hand distance, 
   angles, and lighting.
5. Data is saved as .data and .metadata files in the ./data/ folder.

Model Training
--------------
The classification model is trained using the 'trainml.ipynb' notebook:

1. Preprocessing: The training pipeline includes a custom 
   transformer that normalizes landmarks relative to the wrist 
   (Point 0) and scales them based on the hand size. This ensures 
   the model is distance-invariant.
2. Training: The notebook uses an XGBClassifier to learn the 
   patterns of the 21 hand landmarks (63 total features).
3. Export: The final trained pipeline is saved to 
   './model/classifier.model' using joblib.
4. Deployment: Ensure the generated .model file is in the 
   correct directory for '__main__.py' to load it on startup.

Technical Controls
------------------
* [F]: Toggle Fullscreen
* [P]: Pause Game
* [H]: Return to Home/Reset Match
* [Q]: Quit Application

Shield, Gun, Bullet: Gameplay Documentation
===========================================

Overview
--------
Shield, Gun, Bullet is a two-player, gesture-controlled combat game. 
Players must manage their ammunition and time their attacks while 
maintaining a strict rhythm established by the game's chant phase.

Hand Gestures
-------------
The computer vision system recognizes the following three gestures:

1. Shield: Open palm with fingers pressed together (thumb can point 
   up or be tucked). The player may face either the palm or the 
   back of the hand toward the camera. Use this to block shots.

2. Gun: Index and middle fingers extended and pressed together; 
   pinky and ring fingers closed. The thumb can point up or be 
   tucked. Use this to fire at the opponent (requires 1 bullet).

3. Bullet: A closed fist (thumb can point up or be tucked). Use 
   this to reload (+1 bullet).

Round Structure
---------------
Each round follows a precise state machine to ensure fair play:

1. Countdown: The game begins with a "READY - SET - GO" sequence.
2. Chant Phase: The screen flashes "Shield... Gun... Bullet..." 
   in half-second intervals to synchronize the players' timing.
3. Reveal Phase: During the "Show your hands!" window, the camera 
   activates to capture gestures.
4. Resolution: The system identifies the gestures and updates 
   the HUD.
5. Result: The outcome of the exchange is displayed before the 
   next round begins.

Tactical Rules
--------------
* Critical Hit: If you fire (Gun) while your opponent is 
  reloading (Bullet), you win immediately.
* Ammunition: You cannot fire without bullets. Attempting to 
  use the Gun gesture with 0 bullets results in a warning.
* Shield Fatigue: To prevent defensive camping, using the Shield 
  more than 5 times in a row results in a warning.
* Penalty Loss: Any player who accumulates 3 warnings (from 
  ammo errors or shield fatigue) loses the match.
* Draw: If both players use the same gesture or if a shot is 
  successfully blocked by a shield, the match continues.

Technical Controls
------------------
* [F]: Toggle Fullscreen
* [P]: Pause Game
* [H]: Home/Reset
* [Q]: Quit
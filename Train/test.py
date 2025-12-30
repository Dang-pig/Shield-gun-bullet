import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import joblib as jl

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
class transformer:
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = x.copy()
        x = x.reshape(-1, 21, 3)
        x -= x[:, 0:1, :]
        ref = x[:, 9, :]
        refLen = np.linalg.norm(ref, axis=1, keepdims=True)
        x /= refLen[:, None, :]

        return x.reshape(-1, 63)

# settings
MODELDIR = './model/'
MODELPATH = MODELDIR + "classifier.model"
classModel = jl.load(MODELPATH)
symbolList = ['shield', 'gun', 'bullet', 'unknown']

# cv capture
cap = cv.VideoCapture(0)                # The camera
if not cap.isOpened():                  # Check if the camera is available
    print("Cannot open camera")
    exit()
cv.namedWindow("cam", cv.WINDOW_NORMAL) # Create the display window
cv.resizeWindow("cam", 800, 600)        # Set the width and height
font = cv.FONT_HERSHEY_COMPLEX          # Some font

# Downscaling function
def downscale(frame, scale):
    h, w, c = frame.shape
    h = h - h % scale
    w = w - w % scale

    newframe = frame[:h, :w, :].reshape(h // scale, scale, w // scale, scale, -1).mean(axis=(1, 3)).astype(np.uint8)
    return newframe

# hand landmarks model
modelPath = MODELDIR + "hand_landmarker.task" # The weights and things for the model
BaseOptions = mp.tasks.BaseOptions                                   # The setup for the model (low-level)
HandLandmarker = mp.tasks.vision.HandLandmarker                     # The model code (has detect function)
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions       # Anoher setup specific to HandLandmarker model
options = HandLandmarkerOptions(                                    # Actually create the options
    base_options=BaseOptions(model_asset_path=modelPath),           # Pass the base options
    running_mode=mp.tasks.vision.RunningMode.IMAGE,                 # Pass the running mode (There are IMAGE, VIDEO, and LIVESTREAM. We are using IMAGE here)
    num_hands=2
)

# Show preview and saving data
lastTxt = ""
scal = 1

with HandLandmarker.create_from_options(options) as ldmk: # Create the model here
    while True:
        ret, frame = cap.read()                                         # Read the frame
        if not ret:                                                     # Check if it can read
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)                # Convert opencv image to numpy array, from BGR to RGB
        frame_rgb = downscale(frame_rgb, scal)                             # Downscale the image

        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)    # Have to convert to mediapipe Image type
        res = ldmk.detect(mpImage)                                              # Detection result
        
        # Detection result has 3 attributes:
        #  handedness: a list of left hand or right hand
        #  hand_landmarks: a list of lists of 0 - 1 range scaled position of the landmarks
        #  hand_world_landmarks: distance to the camera or sth (dont care)

        os.system("cls")

        lastTxt = ""
        for i, hand in enumerate(res.hand_landmarks, 0):
            if len(hand) > 0:
                print(f"{res.handedness[i][0].score * 100:.4f}% {res.handedness[i][0].display_name} hand")
                minx, maxx, miny, maxy = 999999, -999999, 999999, -999999
                for id in range(21):
                    xScaled = hand[id].x
                    yScaled = hand[id].y
                    xRiel = int(xScaled * frame_rgb.shape[1])
                    yRiel = int(yScaled * frame_rgb.shape[0])
                    minx = min(minx, xRiel)
                    maxx = max(maxx, xRiel)
                    miny = min(miny, yRiel)
                    maxy = max(maxy, yRiel)
                    cv.circle(frame_rgb, (xRiel, yRiel), 5//scal, (255, 0, 0), -1)
                cv.rectangle(frame_rgb, (minx, miny), (maxx, maxy), (255, 0, 0), 1)

                fea = [f for land in hand for f in [land.x, land.y, land.z]]
                fea = np.array(fea).reshape(1, -1)
                pred = classModel.predict(fea)[0]
                print(pred, symbolList[pred])
                textx = int((minx - 0.1) * frame_rgb.shape[1])
                texty = int((miny - 0.1) * frame_rgb.shape[0])
                cv.putText(frame_rgb, lastTxt, (textx, texty), cv.FONT_HERSHEY_COMPLEX, 1/scal, (255, 0, 255), 1)

        cv.imshow("cam", cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))      # Show the image on the display window (named "cam")
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
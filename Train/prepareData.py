import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# settings
DATADIR = "./data/"
os.makedirs(DATADIR, exist_ok=True)

imCnt = 0
SYMBOL = "unknown"
TESTFLAG = True
TEST = '_test' if TESTFLAG else ''
METADATA = DATADIR + SYMBOL + TEST + ".metadata"
DATAPATH = DATADIR + SYMBOL + TEST + ".data"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

try:
    with open(METADATA, "r") as f:
        imCnt = f.readline()
        print(f"Read {imCnt}")
        if not is_number(imCnt):
            imCnt = 0
            print("Can't read a numeric value")
        imCnt = int(imCnt)
        print("Loaded metadata", imCnt)
except:
    print("Metadata file doesn't exist or sth")

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
modelPath = "./../model/hand_landmarker.task" # The weights and things for the model
BaseOptions = mp.tasks.BaseOptions                                   # The setup for the model (low-level)
HandLandmarker = mp.tasks.vision.HandLandmarker                     # The model code (has detect function)
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions       # Anoher setup specific to HandLandmarker model
options = HandLandmarkerOptions(                                    # Actually create the options
    base_options=BaseOptions(model_asset_path=modelPath),           # Pass the base options
    running_mode=mp.tasks.vision.RunningMode.IMAGE                  # Pass the running mode (There are IMAGE, VIDEO, and LIVESTREAM. We are using IMAGE here)
)

lastTxt = f"There has been {imCnt} entries so far"
# Show preview and saving data
scal = 3
with open(DATAPATH, "a") as f, HandLandmarker.create_from_options(options) as ldmk: # Create the model here
    while True:
        ret, frame = cap.read()                                         # Read the frame
        if not ret:                                                     # Check if it can read
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)                # Convert opencv image to numpy array, from BGR to RGB
        frame_rgb = downscale(frame_rgb, scal)                             # Downscale the image

        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)    # Have to convert to mediapipe Image type
        res = ldmk.detect(mpImage)                                              # Detection result
        
        # Detection result has 3 attributes:
        #  handedness: left hand or right hand
        #  hand_landmarks: 0 - 1 range scaled position of the landmarks
        #  hand_world_landmarks: distance to the camera or sth (dont care)

        os.system("cls")
        hashand = 1

        if len(res.hand_landmarks) > 0:
            print("Hand detected")
            print(f"{res.handedness[0][0].score * 100:.4f}% {res.handedness[0][0].display_name} hand")
            for i in range(21):
                xScaled = res.hand_landmarks[0][i].x
                yScaled = res.hand_landmarks[0][i].y
                xRiel = int(xScaled * frame_rgb.shape[1])
                yRiel = int(yScaled * frame_rgb.shape[0])
                cv.circle(frame_rgb, (xRiel, yRiel), 5//scal, (255, 0, 0), -1)
        else:
            hashand = 0
            print("No hand")

        print(lastTxt)
        cv.imshow("cam", cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))      # Show the image on the display window (named "cam")

        key = cv.waitKey(1) & 0xFF

        if key == ord('c') and hashand:
            imCnt += 1
            line = ''
            for i, mark in enumerate(res.hand_landmarks[0], 0):
                line += str(mark.x) + "," + str(mark.y) + ',' + str(mark.z) + ','
            f.write(line + "\n")
            lastTxt = f"Saved entry #{imCnt} to {DATAPATH}"
        elif key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

with open(METADATA, "w") as f:
    f.write(str(imCnt))
    f.write("\n")
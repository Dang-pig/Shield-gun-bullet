import cv2 as cv
import numpy as np

# Settings
RED = (255, 0, 0)
ORANGE = (255, 160, 25)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (165, 0, 255)
PINK = (255, 0, 255)
BLACK = (0, 0, 0)
DARKGREY = (20, 20, 20)
GREY = (50, 50, 50)
LIGHTGREY = (80, 80, 80)
WHITE = (255, 255, 255)
LIGHTBLUE = (55, 125, 255)

DEFAULTFONT = cv.FONT_HERSHEY_COMPLEX

FULLSCREEN = False
QUIT = 690
OK = 691
NOCAM = 692
PAUSE = False

# App states:
HOME = 69690
INST = 69691
SETT = 69692
GAME = 69693
# In game state:
READY = 6969690
SET = 6969691
GO = 6969692
PREPARE1 = 6969693
PREPARE2 = 6969694
PREPARE3 = 6969695
WAITFORCAM = 6969696
INGAME = 6969697
SHOWRESULT = 6969698
ENDGAME = 6969699

APPSTATE = HOME
INGAMESTATE = None
ROUNDSTATE = None
GAMETIMECHECKPOINT = 0.0
TIMEFROMCHECKPOINT = 0.0
LASTTICK = 0.0
camFrame = None
LAST_ROUND_RESULT = ""  # Track the last round result for home screen
handRes = None

# Some lists
allObjects = []
hoverCheck = []
renderList = []

# All elements' widths and heights will be set as reference to window's width and height
windowWidth = 1280
windowHeight = 960
print(windowWidth, windowHeight)

# Classes and backend
def convert(widthRef, heightRef):
    return int(windowWidth * widthRef), int(windowHeight * heightRef)

class Object():
    def __init__(self, *, name,
                onClick=None, onHoverEnter=None, onHoverExit=None):
        self.name = name
        self.hovered = False
        self.shown = True
        self.child = []
        self.z = 0

        self.onClick = onClick
        self.onHoverEnter = onHoverEnter
        self.onHoverExit = onHoverExit

        global allObjects
        allObjects.append(self)
        allObjects = sorted(allObjects, key=lambda ob:ob.z)
    
    def render(self, img):
        if not self.shown:
            return
        if hasattr(self, 'selfRender'):
            self.selfRender(img)

    def show(self):
        self.shown = True
    
    def hide(self):
        self.shown = False
    
    def appendChild(self, ob):
        self.child.append(ob)
        self.child = sorted(self.child, key=lambda x: x.z)
    
    def getChild(self, name):
        for ob in self.child:
            if ob.name == name:
                return ob

class Box(Object):
    def __init__(self, *, name="", x, y, z=0, width, height, color=BLACK, centered=True, fill=True, lineWidth=None,
                onClick=None, onHoverEnter=None, onHoverExit=None):
        super().__init__(name=name,
                        onClick=onClick, onHoverEnter=onHoverEnter, onHoverExit=onHoverExit
                        )
        self.baseWidth = width
        self.rielWidth = width
        self.baseHeight = height
        self.rielHeight = height
        self.color = color
        self.x = x
        self.y = y
        self.z = z
        self.centered = centered
        self.fill = fill
        self.baseLineWidth = lineWidth
        self.rielLineWidth = lineWidth
        self.lineThicknessParam = -1 if self.fill else self.rielLineWidth
    
    def selfRender(self, img):
        (xpos, ypos) = convert(self.x - self.rielWidth / 2, self.y - self.rielHeight / 2) if(self.centered) else convert(self.x, self.y)
        (wid, hei) = convert(self.rielWidth, self.rielHeight)
        cv.rectangle(img, (xpos, ypos), (xpos + wid, ypos + hei), self.color, self.lineThicknessParam)
        
    def __getBoudingBox__(self):
        lf = self.x - (self.rielWidth / 2 if self.centered else 0)
        up = self.y - (self.rielHeight / 2 if self.centered else 0)
        return (
            lf,
            up,
            lf + self.rielWidth,
            up + self.rielHeight
        )
    
    def __hoverCheck__(self, mx, my):
        lf, up, rt, dn = self.__getBoudingBox__()
        
        self.hovered = (lf <= mx <= rt and up <= my <= dn)
        return self.hovered and self.shown
    
    def scale(self, ratio):
        self.rielWidth = self.baseWidth * ratio
        self.rielHeight = self.baseHeight * ratio
        self.rielLineWidth = self.baseLineWidth * ratio
        self.lineThicknessParam = -1 if self.fill else self.rielLineWidth
        
    def show(self):
        self.shown = True

class Text(Object):
    def __init__(self, *, name="", x, y, z=0, color=BLACK, content="", fontScale=1.0, centered=True, lineWidth=1,
                onClick=None, onHoverEnter=None, onHoverExit=None):
        super().__init__(name=name,
                        onClick=onClick, onHoverEnter=onHoverEnter, onHoverExit=onHoverExit
                        )
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.content = content
        self.font = DEFAULTFONT
        self.baseFontScale = fontScale
        self.rielFontScale = fontScale
        self.centered = centered
        self.baseLineWidth = lineWidth
        self.rielLineWidth = lineWidth

        (self.width, self.height), self.baseLine = cv.getTextSize(self.content, self.font, self.rielFontScale, self.rielLineWidth)
        self.width /= windowWidth
        self.height /= windowHeight
        self.baseLine /= windowHeight

    
    def selfRender(self, img):
        if not self.shown:
            return
        (xpos, ypos) = convert(self.x - self.width / 2, self.y + self.height / 2) if self.centered else convert(self.x, self.y + self.height)
        cv.putText(img, self.content, (xpos, ypos), self.font, self.rielFontScale, self.color, self.rielLineWidth)
    
    def __reCalWH__(self):
        (self.width, self.height), self.baseLine = cv.getTextSize(self.content, self.font, self.rielFontScale, self.rielLineWidth)
        self.width /= windowWidth
        self.height /= windowHeight
        self.baseLine /= windowHeight

    def changeContent(self, newContent):
        self.content = newContent
        self.__reCalWH__()
    
    def __getBoudingBox__(self):
        lf = self.x - (self.width / 2 if self.centered else 0)
        up = self.y - (self.height / 2 if self.centered else 0)
        return (
            lf,
            up,
            lf + self.width,
            up + self.height
        )
    
    def __hoverCheck__(self, mx, my):
        lf, up, rt, dn = self.__getBoudingBox__()
        
        self.hovered = (lf <= mx <= rt and up <= my <= dn)
        return self.hovered and self.shown
    
    def scale(self, ratio):
        self.rielFontScale = self.baseFontScale * ratio
        self.rielLineWidth = int(self.baseLineWidth * ratio)
        self.__reCalWH__()

class Button(Object):
    def __init__(self, *, name='', x, y, z=0,
                 adaptiveBox=True, boxTextPadding=0, boxWidth=None, boxHeight=None, boxColor=WHITE, boxFill=True, boxLineWidth=1,
                 textColor=BLACK, textContent="", textFontScale=1, textLineWidth=1,
                 centered=True,
                 onClick=None, onHoverEnter=None, onHoverExit=None):
        super().__init__(name=name,
                        onClick=onClick, onHoverEnter=onHoverEnter, onHoverExit=onHoverExit
                        )
        self.x = x
        self.y = y
        self.z = z
        self.centered = centered

        self.__text__ = Text(name=self.name + '_buttonText',
                              x=self.x, y=self.y, z=self.z,
                              color=textColor, content=textContent,
                              fontScale=textFontScale, centered=True, lineWidth=textLineWidth)

        self.padding = boxTextPadding

        textW = self.__text__.width
        textH = self.__text__.height + self.__text__.baseLine

        boxW = boxWidth if not adaptiveBox else textW
        boxH = boxHeight if not adaptiveBox else textH
        boxW += self.padding * 2
        boxH += self.padding * 2
        self.__box__ = Box(name=self.name + '_buttonBox',
                           x = self.x + (boxW / 2 if not self.centered else 0),
                           y = self.y + (boxH / 2 if not self.centered else 0),
                           z=self.z,
                           width=boxW, height=boxH, color=boxColor,
                           centered=True, fill=boxFill, lineWidth=boxLineWidth)

        self.__text__.x = self.x + (boxW / 2 if not self.centered else 0)
        self.__text__.y = self.y + (boxH / 2 if not self.centered else 0)

        self.appendChild(self.__box__)
        self.appendChild(self.__text__)

    def __getBoudingBox__(self):
        lf = self.x - (self.__box__.rielWidth / 2 if self.centered else 0)
        up = self.y - (self.__box__.rielHeight / 2 if self.centered else 0)
        return (
            lf,
            up,
            lf + self.__box__.rielWidth,
            up + self.__box__.rielHeight
        )
    
    def __hoverCheck__(self, mx, my):
        lf, up, rt, dn = self.__getBoudingBox__()
        
        self.hovered = (lf <= mx <= rt and up <= my <= dn)
        return self.hovered and self.shown
    
    def scale(self, ratio):
        self.__box__.scale(ratio)
        self.__text__.scale(ratio)
    
    def show(self):
        self.shown = True
        self.__box__.show()
        self.__text__.show()

    def hide(self):
        self.shown = False
        self.__box__.hide()
        self.__text__.hide()

class Pack(Object):
    def __init__(self, name=''):
        super().__init__(name=name)

def addToHoverCheck(ob):
    global hoverCheck
    hoverCheck.append(ob)
    hoverCheck = sorted(hoverCheck, key=lambda ob:ob.z, reverse=True)

def changeAppState(state):
    global APPSTATE
    APPSTATE = state

def mouseEventHandler(event, mx, my, flags, param):
    mx /= windowWidth
    my /= windowHeight
    if event == cv.EVENT_MOUSEMOVE:
        updateHover(mx, my)
    if event == cv.EVENT_LBUTTONDOWN:
        clicked(mx, my)

def logAll():
    for ob in allObjects:
        print(vars(ob))

def updateHover(mx, my):
    for ob in hoverCheck:
        if not hasattr(ob, '__hoverCheck__'):
            continue
        now = ob.__hoverCheck__(mx, my)
        if now:
            ob.hovered = True
            if ob.onHoverEnter:
                ob.onHoverEnter(ob)
        else:
            ob.hovered = False
            if ob.onHoverExit:
                ob.onHoverExit(ob)

def clicked(mx, my):
    for ob in hoverCheck:
        if ob.hovered and ob.onClick:
            ob.onClick()

def collectRenderObjects(root, queue):
    queue.append(root)
    for ch in getattr(root, 'child', []):
        collectRenderObjects(ch, queue)

# Frontend - Game Elements
frame = np.zeros((windowHeight, windowWidth, 3)).astype(np.uint8)
#Home screen
homePack = Pack()
homePack.appendChild(Box(width=1, height=1, color=LIGHTGREY, x=0, y=0, z=-1, centered=False,))
homePack.appendChild(Text(x=0.5, y=0.2, z=0, color=WHITE, lineWidth=3, fontScale=1.5, content='Shield, Gun, Bullet!',))
homePack.appendChild(Text(name='lastRoundText', x=0.5, y=0.35, z=0, color=YELLOW, lineWidth=2, fontScale=0.8, content='', centered=True))
homePack.appendChild(Button(name='PlayBut', x=0.5, y=0.7, z=0, adaptiveBox=False, boxWidth=0.3, boxHeight=0.08, boxTextPadding=0.005, boxColor=LIGHTBLUE, boxFill=True, textColor=WHITE, textContent='PLAY', textLineWidth=2, centered=True, onHoverEnter=lambda self:self.scale(1.2), onHoverExit=lambda self:self.scale(1)))
homePack.appendChild(Button( name='InstrBut', x=0.5, y=0.85, z=0, adaptiveBox=False, boxWidth=0.3, boxHeight=0.08, boxTextPadding=0.005, boxColor=LIGHTBLUE, boxFill=True, textColor=WHITE, textContent='Instruction', textLineWidth=2, textFontScale=0.8, centered=True, onHoverEnter=lambda self:self.scale(1.2), onHoverExit=lambda self:self.scale(1), onClick=lambda:changeAppState(INST),))
addToHoverCheck(homePack.getChild('PlayBut'))
addToHoverCheck(homePack.getChild('InstrBut'))
#Instruction screen
instrPack = Pack()
instrPack.appendChild(Box(
    x=0, y=0, z=-1,
    width=1, height=1,
    color=LIGHTGREY,
    centered=False
))
instrLine = [
    'Press PLAY to play',
    "Each player's zone is a half of the screen",
    'There will be multiple rounds, each round will be like this:',
    ' - First there will be a time to prepare (1 second or so)',
    ' - The camera appears, each player shows their hand',
    ' - The game will predict your hand symbol',
    ' - If you choose bullet, your bullet count + 1',
    ' - If you choose gun, you shoot (Only when you have bullets)',
    ' - If you choose shield, you protect yourself from the gun',
    ' - You lose when the other shoot while you are holding a bullet',
    ' - If you shoot with no bullets or shiled more than 5 times, you will be warned',
    ' - Being warned more than 3 times will make you lose the game',
    ' - The bullet count and warning count is at the bottom',
    '   (B = bullet warning, S = shield warning)'
]
for i, txt in enumerate(instrLine, 0):
    instrPack.appendChild(Text(x=0.06, y=0.15 + 0.03 * i, color=WHITE, content=txt, fontScale=0.5, lineWidth=1, centered=False))
#Game screen
gamePack = Pack()
#Back to home button
homeButton = Button(x=0.02, y=0.02,z=2, adaptiveBox=False, boxTextPadding=0.005, boxColor=LIGHTBLUE, boxWidth=0.05, boxHeight=0.03, textColor=WHITE, textContent='Home', textFontScale=0.5, textLineWidth=1, centered=False, onHoverEnter=lambda self:self.scale(1.1), onHoverExit=lambda self:self.scale(1))
addToHoverCheck(homeButton)
#Paused screen
pausePack = Pack()
pausePack.appendChild(Box(x=0, y=0, z=999-1, width=1, height=1, color=GREY, centered=False))
pausePack.appendChild(Text(x=0.5, y=0.5, z=999, color=WHITE, content='Paused', fontScale=1.6, lineWidth=3, centered=True))
#End game screen
endGamePack = Pack()
endGamePack.appendChild(Box(x=0, y=0, z=5, width=1, height=1, color=BLACK, centered=False))
endGamePack.appendChild(Text(name='endGameText', x=0.5, y=0.5, z=6, color=WHITE, content='GAME OVER', fontScale=2.0, lineWidth=4, centered=True))
#Refresh box
refresh = Box(x=0,y=0,z=-99999, width=1, height=1, color=BLACK, centered=False)

cap = cv.VideoCapture(0)

def resetPlayers():
    """Reset all player states and UI elements"""
    for player in players:
        player["bulletCount"] = 0
        player["bulletWarnStreak"] = 0
        player["bulletWarnCount"] = 0
        player["shieldUseStreak"] = 0
        player["shieldWarnCount"] = 0
        # Reset UI text elements
        player["bulletText"].changeContent("Bullets: 0")
        player["warnText"].changeContent("Warn B: 0 | S: 0")

def goHome():
    resetPlayers()  # Reset players when returning to home
    changeAppState(HOME)

homeButton.onClick = goHome

def drawCamera(img):
    global camFrame, handRes
    ret, cam = cap.read()
    if not ret:
        return NOCAM, None
    cam = cv.cvtColor(cam, cv.COLOR_BGR2RGB).astype(np.uint8)
    # Keep a clean copy of the camera frame for the hand model (no UI drawn on top)
    camFrame = cam.copy()
    handRes = handModel.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=camFrame))
    for marks in handRes.hand_landmarks:
        for mark in marks:
            renderList.append(Box(x=mark.x, y=mark.y, z=3, width=0.01, height=0.01, color=GREEN))

    cam = cv.resize(cam, (windowWidth, windowHeight), interpolation=cv.INTER_LINEAR)
    camShape = cam.shape
    imgShape = img.shape
    img[:min(camShape[0], imgShape[0]), :min(camShape[1], imgShape[1]), :] = cam[:min(camShape[0], imgShape[0]), :min(camShape[1], imgShape[1]), :]
    return OK, img

gameTimeLog = Text(x=0.5, y=0.1, z=99, color=RED,content='', centered=True, fontScale=0.7, lineWidth=1)

def startGame():
    global INGAMESTATE, LASTTICK, TIMEFROMCHECKPOINT, GAMETIMECHECKPOINT
    changeAppState(GAME)
    GAMETIMECHECKPOINT = cv.getTickCount()
    LASTTICK = GAMETIMECHECKPOINT
    TIMEFROMCHECKPOINT = 0.0
    # Reset players for new game
    resetPlayers()
    # Show READY / SET / GO once before any round happens
    INGAMESTATE = READY

homePack.getChild('PlayBut').onClick = startGame

# READY / SET / GO screens (shown once at game start)
readyPack = Pack()
readyPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.4, height=0.3, color=GREEN))
readyPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.7, content='READY', lineWidth=2))

setPack = Pack()
setPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.4, height=0.3, color=GREEN))
setPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.7, content='SET', lineWidth=2))

goPack = Pack()
goPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.4, height=0.3, color=GREEN))
goPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.7, content='GO', lineWidth=2))

# Round "chant" screens for each round – like saying Rock, Paper, Scissors
prepareShieldPack = Pack()
prepareShieldPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.2, height=0.1, color=LIGHTBLUE))
prepareShieldPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.3, content='Shield', lineWidth=2))

prepareGunPack = Pack()
prepareGunPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.2, height=0.1, color=LIGHTBLUE))
prepareGunPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.3, content='Gun', lineWidth=2))

prepareBulletPack = Pack()
prepareBulletPack.appendChild(Box(x=0.5, y=0.5, z=5, width=0.2, height=0.1, color=LIGHTBLUE))
prepareBulletPack.appendChild(Text(x=0.5, y=0.5, z=6, fontScale=1.3, content='Bullet', lineWidth=2))

waitForCamPack = Pack()
waitForCamPack.appendChild(Box(x=0.5, y=0.2, z=5, width=0.2, height=0.1, color=LIGHTBLUE))
waitForCamPack.appendChild(Text(x=0.5, y=0.2, z=6, fontScale=0.6, content='Show your hands!', lineWidth=2))

#Split screen line
splitScreen = Box(x=0.5, y=0.5, z=4, width=0.005, height=1, color=LIGHTBLUE)

# Models here
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

MODELDIR = './model/'

CLASSMODELPATH = MODELDIR + "classifier.model"
classModel = jl.load(CLASSMODELPATH)
SYMBOLLIST = ['shield', 'gun', 'bullet', 'unknown']
SHIELD, GUN, BULLET, UNKNOWN = 0, 1, 2, 3

HANDMODELPATH = MODELDIR + "hand_landmarker.task"                       # The weights and things for the model
BASEOPTION = mp.tasks.BaseOptions                                       # The setup for the model (low-level)
HANDMODELCLASS = mp.tasks.vision.HandLandmarker                         # The model code (has detect function)
HANDMODELOPTION = mp.tasks.vision.HandLandmarkerOptions                 # Anoher setup specific to HandLandmarker model
# Actually create the options for the HandLandmarker model
OPTIONS = HANDMODELOPTION(
    base_options=BASEOPTION(model_asset_path=HANDMODELPATH),            # Pass the base options
    running_mode=mp.tasks.vision.RunningMode.IMAGE,                     # Pass the running mode (There are IMAGE, VIDEO, and LIVESTREAM. We are using IMAGE here)
    num_hands=2
)

# Single shared hand landmark model and classifier for both players
handModel = HANDMODELCLASS.create_from_options(OPTIONS)

# Per-player UI and state, stored in an array to avoid duplication
players = []

# Player 1 (left)
player1BulletText = Text(
    name="p1BulletText",
    x=0.25, y=0.9, z=10,
    color=YELLOW,
    content='Bullets: 0',
    fontScale=0.8,
    centered=True,
    lineWidth=2
)
player1WarnText = Text(
    name="p1WarnText",
    x=0.25, y=0.95, z=10,
    color=ORANGE,
    content='Warn B: 0 | S: 0',
    fontScale=0.6,
    centered=True,
    lineWidth=1
)
player1ResultPack = Pack(name='p1ResultPack')
player1ResultPack.appendChild(Box(
    name='p1ResultBg',
    x=0.25, y=0.5, z=8,
    width=0.3, height=0.15,
    color=LIGHTGREY,
    centered=True,
    fill=True,
    lineWidth=1
))
player1ResultPack.appendChild(Text(
    name='p1ResultText',
    x=0.25, y=0.5, z=9,
    color=WHITE, content='', fontScale=0.8, centered=True, lineWidth=2
))
players.append({
    "bulletCount": 0,
    "bulletWarnStreak": 0,
    "bulletWarnCount": 0,
    "shieldUseStreak": 0,
    "shieldWarnCount": 0,
    "bulletText": player1BulletText,
    "warnText": player1WarnText,
    "resultPack": player1ResultPack,
    "resultTextName": "p1ResultText",
})

# Player 2 (right)
player2BulletText = Text(
    name="p2BulletText",
    x=0.75, y=0.9, z=10,
    color=YELLOW,
    content='Bullets: 0',
    fontScale=0.8,
    centered=True,
    lineWidth=2
)
player2WarnText = Text(
    name="p2WarnText",
    x=0.75, y=0.95, z=10,
    color=ORANGE,
    content='Warn B: 0 | S: 0',
    fontScale=0.6,
    centered=True,
    lineWidth=1
)
player2ResultPack = Pack(name='p2ResultPack')
player2ResultPack.appendChild(Box(
    name='p2ResultBg',
    x=0.75, y=0.5, z=8,
    width=0.3, height=0.15,
    color=LIGHTGREY,
    centered=True,
    fill=True,
    lineWidth=1
))
player2ResultPack.appendChild(Text(
    name='p2ResultText',
    x=0.75, y=0.5, z=9,
    color=WHITE, content='', fontScale=0.8, centered=True, lineWidth=2
))
players.append({
    "bulletCount": 0,
    "bulletWarnStreak": 0,
    "bulletWarnCount": 0,
    "shieldUseStreak": 0,
    "shieldWarnCount": 0,
    "bulletText": player2BulletText,
    "warnText": player2WarnText,
    "resultPack": player2ResultPack,
    "resultTextName": "p2ResultText",
})

def handleRoundLogic(img):
    global players, camFrame, handRes

    if camFrame is None:
        # No camera frame yet, just ask both players to show hands
        for player in players:
            playerResultText = player["resultPack"].getChild(player["resultTextName"])
            playerResultText.changeContent("Please show your hand")
        return

    symbolIdxs = [None, None]
    hasHandFlags = [False, False]

    numLandmarks = 21
    
    lmk = [[], []]

    for i, hand in enumerate(handRes.hand_landmarks):
        Xs = [mark.x for mark in hand]
        minX = min(Xs)
        maxX = max(Xs)
        isPlayer = (1 if minX >= 0.5 else (0 if maxX < 0.5 else -1))
        if isPlayer == -1:
            continue
        lmk[isPlayer] = [{'x':mark.x, 'y':mark.y, 'z':mark.z} for mark in hand]
        hasHandFlags[isPlayer] = True

    for i in range(2):
        if hasHandFlags[i]:
            fea = [val for d in lmk[i] for val in [d['x'], d['y'], d['z']]]
            fea = np.array(fea).reshape(1, -1)
            predIdx = int(classModel.predict(fea)[0])
            symbolIdxs[i] = predIdx
            print(i, "has hand")
            print(fea)
            print(classModel.predict_proba(fea)[0])
            print(predIdx)

    def applySymbol(symbolIdx, playerIdx, bulletAdd, warningAdd, bulletChange, shieldChange):
        actionDesc = ''
        bulletWarnedThisRound = False
        shieldWarnedThisRound = False
        lost = False

        if symbolIdx == BULLET:
            actionDesc = 'Reload!'
            bulletAdd[playerIdx] = 1  # Add 1 bullet
            warningAdd[playerIdx] = 0  # No warning
            bulletChange[playerIdx] = 0  # Reset streak
            shieldChange[playerIdx] = 0  # Reset streak
        elif symbolIdx == GUN:
            shieldChange[playerIdx] = 0  # Reset shield streak
            warningAdd[playerIdx] = 0  # Reset shield warnings for gun
            if prevBulletCounts[playerIdx] > 0:
                bulletAdd[playerIdx] = -1  # Remove 1 bullet (shooting)
                actionDesc = 'Shoot!'
                bulletChange[playerIdx] = 0  # Reset bullet streak
            else:
                actionDesc = 'No bullets!'
                bulletWarnedThisRound = True
                bulletChange[playerIdx] = 1  # Increment streak
                warningAdd[playerIdx] = 1  # Add bullet warning
                if players[playerIdx]["bulletWarnStreak"] + 1 >= 3:
                    lost = True
        elif symbolIdx == SHIELD:
            actionDesc = 'Block!'
            bulletChange[playerIdx] = 0  # Reset bullet streak
            shieldChange[playerIdx] = 1  # Increment shield streak
            if players[playerIdx]["shieldUseStreak"] + 1 > 5:
                warningAdd[playerIdx] = 1  # Add shield warning
                actionDesc = 'Too many shields!'
                shieldWarnedThisRound = True
                if players[playerIdx]["shieldWarnCount"] + 1 > 3:
                    lost = True
        elif symbolIdx == UNKNOWN:
            actionDesc = "Sorry, I don't understand"
            bulletChange[playerIdx] = 0  # Reset bullet streak
            shieldChange[playerIdx] = 0  # Reset shield streak
            warningAdd[playerIdx] = 0  # No shield warning
        else:
            actionDesc = SYMBOLLIST[symbolIdx]
            bulletChange[playerIdx] = 0  # Reset bullet streak
            shieldChange[playerIdx] = 0  # Reset shield streak
            warningAdd[playerIdx] = 0  # No shield warning

        return actionDesc, lost, bulletWarnedThisRound, shieldWarnedThisRound

    prevBulletCounts = [p["bulletCount"] for p in players]

    # Initialize arrays to collect changes
    bulletAdd = [0, 0]  # How many bullets to add/remove for each player
    warningAdd = [0, 0]  # How many warnings to add for each player
    bulletChange = [0, 0]  # How to change bullet streak (0=reset, 1=increment)
    shieldChange = [0, 0]  # How to change shield streak (0=reset, 1=increment)

    results = []
    anyWarnedThisRound = False
    bothPlayersValid = True  # Track if both players have valid gestures

    for i, player in enumerate(players):
        if not hasHandFlags[i] or symbolIdxs[i] is None:
            results.append({
                "actionDesc": "Please show your hand",
                "lost": False,
            })
            bothPlayersValid = False  # If any player doesn't have valid gesture, mark as invalid
            continue

        actionDesc, lost, bulletWarnedThisRound, shieldWarnedThisRound = applySymbol(
            symbolIdxs[i], i, bulletAdd, warningAdd, bulletChange, shieldChange
        )

        anyWarnedThisRound = anyWarnedThisRound or bulletWarnedThisRound or shieldWarnedThisRound

        results.append({
            "actionDesc": actionDesc,
            "lost": lost,
        })

    # If ANY player has UNKNOWN symbol, zero out bullet and warning additions for BOTH players
    anyPlayerUnknown = any(
        (hasHandFlags[i] and symbolIdxs[i] == UNKNOWN)
        for i in range(2)
    )

    if anyPlayerUnknown:
        bulletAdd = [0, 0]
        warningAdd = [0, 0]

    # Apply changes based on validity
    for i, player in enumerate(players):
        if not hasHandFlags[i] or symbolIdxs[i] is None:
            continue

        # Always apply warnings and streaks
        if bulletChange[i] == 0:
            player["bulletWarnStreak"] = 0
        else:
            player["bulletWarnStreak"] += bulletChange[i]

        if shieldChange[i] == 0:
            player["shieldUseStreak"] = 0
        else:
            player["shieldUseStreak"] += shieldChange[i]

        player["bulletWarnCount"] += warningAdd[i]
        player["shieldWarnCount"] += warningAdd[i]

        # Only apply bullet changes if both players have valid gestures AND no warnings this round
        if bothPlayersValid and not anyWarnedThisRound:
            player["bulletCount"] += bulletAdd[i]

        # Update bullet HUD
        player["bulletText"].changeContent(f"Bullets: {player['bulletCount']}")
        # Update warning HUD (B: bullet warns, S: shield warns)
        player["warnText"].changeContent(
            f"Warn B: {player['bulletWarnCount']} | S: {player['shieldWarnCount']}"
        )

    # Check for shooting vs bullet holding mechanic
    # If one player shoots (valid) while the other holds bullet, the bullet player loses
    p1Shot = (hasHandFlags[0] and symbolIdxs[0] == GUN and prevBulletCounts[0] > 0)  # Player 1 successfully shot
    p2Shot = (hasHandFlags[1] and symbolIdxs[1] == GUN and prevBulletCounts[1] > 0)  # Player 2 successfully shot
    p1HeldBullet = (hasHandFlags[0] and symbolIdxs[0] == BULLET)  # Player 1 chose bullet this round
    p2HeldBullet = (hasHandFlags[1] and symbolIdxs[1] == BULLET)  # Player 2 chose bullet this round

    if p1Shot and p2HeldBullet:
        results[1]["lost"] = True  # Player 2 loses for holding bullet while shot at
    if p2Shot and p1HeldBullet:
        results[0]["lost"] = True  # Player 1 loses for holding bullet while shot at

    p1Action = results[0]["actionDesc"]
    p2Action = results[1]["actionDesc"]
    p1Lost = results[0]["lost"]
    p2Lost = results[1]["lost"]

    # Set last round result for home screen
    global LAST_ROUND_RESULT

    p1ResultText = players[0]["resultPack"].getChild(players[0]["resultTextName"])
    p2ResultText = players[1]["resultPack"].getChild(players[1]["resultTextName"])

    if p1Lost and p2Lost:
        p1ResultText.changeContent('Draw: both lose!')
        p2ResultText.changeContent('Draw: both lose!')
        endGamePack.getChild('endGameText').changeContent('DRAW!')
        LAST_ROUND_RESULT = "Last round: DRAW"
    elif p1Lost:
        p1ResultText.changeContent('You lose!')
        p2ResultText.changeContent('You win!')
        endGamePack.getChild('endGameText').changeContent('PLAYER 2 WIN!')
        LAST_ROUND_RESULT = "Last round: PLAYER 2 WIN"
    elif p2Lost:
        p1ResultText.changeContent('You win!')
        p2ResultText.changeContent('You lose!')
        endGamePack.getChild('endGameText').changeContent('PLAYER 1 WIN')
        LAST_ROUND_RESULT = "Last round: PLAYER 1 WIN"
    else:
        p1ResultText.changeContent(f'{p1Action}')
        p2ResultText.changeContent(f'{p2Action}')
        LAST_ROUND_RESULT = ""

def handleGame(img):
    global INGAMESTATE, LASTTICK, TIMEFROMCHECKPOINT, GAMETIMECHECKPOINT
    now = cv.getTickCount()
    dt = (now - LASTTICK) / cv.getTickFrequency()
    TIMEFROMCHECKPOINT += dt
    LASTTICK = now

    gameTimeLog.changeContent(f'Time from checkpoint: {TIMEFROMCHECKPOINT:.2f}s')
    renderList.append(gameTimeLog)

    readyStates = [READY, SET, GO]
    readyPacks = [readyPack, setPack, goPack]
    readyNext = [SET, GO, PREPARE1]

    if INGAMESTATE in readyStates and TIMEFROMCHECKPOINT >= 1.0:
            idx = readyStates.index(INGAMESTATE)
            GAMETIMECHECKPOINT = now
            TIMEFROMCHECKPOINT = 0.0
            INGAMESTATE = readyNext[idx]

    if INGAMESTATE in readyStates:
        renderList.append(readyPacks[readyStates.index(INGAMESTATE)])
        return

    renderList.append(splitScreen)
    # Per-player HUD: bullet counters and warning counters
    for player in players:
        renderList.append(player["bulletText"])
        renderList.append(player["warnText"])
    prepare_states = [PREPARE1, PREPARE2, PREPARE3]
    prepare_packs = [prepareShieldPack, prepareGunPack, prepareBulletPack]
    prepareNext = [PREPARE2, PREPARE3, WAITFORCAM]

    if INGAMESTATE in prepare_states and TIMEFROMCHECKPOINT >= 0.5:
        idx = prepare_states.index(INGAMESTATE)
        GAMETIMECHECKPOINT = now
        TIMEFROMCHECKPOINT = 0.0
        INGAMESTATE = prepareNext[idx]

    if INGAMESTATE in prepare_states:
        renderList.append(prepare_packs[prepare_states.index(INGAMESTATE)])
        return

    if INGAMESTATE == WAITFORCAM:
        renderList.append(waitForCamPack)
        if TIMEFROMCHECKPOINT >= 1:
            GAMETIMECHECKPOINT = now
            TIMEFROMCHECKPOINT = 0.0
            INGAMESTATE = INGAME
        return
    
    if INGAMESTATE == INGAME:
        handleRoundLogic(img)
        GAMETIMECHECKPOINT = now
        TIMEFROMCHECKPOINT = 0.0
        INGAMESTATE = SHOWRESULT
        return

    if INGAMESTATE == SHOWRESULT:
        # Show each player's result pack
        for player in players:
            renderList.append(player["resultPack"])
        if TIMEFROMCHECKPOINT >= 1.2:
            GAMETIMECHECKPOINT = now
            TIMEFROMCHECKPOINT = 0.0
            # Check if someone lost this round - if so, end the game
            p1Lost = players[0]["resultPack"].getChild(players[0]["resultTextName"]).content.endswith("lose!")
            p2Lost = players[1]["resultPack"].getChild(players[1]["resultTextName"]).content.endswith("lose!")
            if p1Lost or p2Lost:
                INGAMESTATE = ENDGAME
            else:
                INGAMESTATE = PREPARE1
        return

    if INGAMESTATE == ENDGAME:
        renderList.append(endGamePack)
        if TIMEFROMCHECKPOINT >= 2.0:  # Show end game for 2 seconds
            goHome()
        return

def handleKey(key):
    global FULLSCREEN, PAUSE
    if key == ord('q'):
        return QUIT
    if key == ord('f'):
        FULLSCREEN = not FULLSCREEN
        cv.setWindowProperty(windowName, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN if FULLSCREEN else cv.WINDOW_NORMAL)
    if key == ord('p'):
        PAUSE = not PAUSE
    if key == ord('h'):
        goHome()

# Init window
windowName = 'Shield Gun Bullet'
app = cv.namedWindow(windowName, cv.WINDOW_NORMAL)
cv.resizeWindow(windowName, windowWidth, windowHeight)

# Main loop
cv.setMouseCallback(windowName, mouseEventHandler)

if __name__ == '__main__':
    while cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) > 0:
        refresh.render(frame)
        renderList = []
        if not PAUSE:
            if APPSTATE == HOME:
                # Update last round result text
                lastRoundText = homePack.getChild('lastRoundText')
                lastRoundText.changeContent(LAST_ROUND_RESULT)
                renderList.append(homePack)
            elif APPSTATE == INST:
                renderList.append(instrPack)
                renderList.append(homeButton)
            elif APPSTATE == GAME:
                exitcode, frame = drawCamera(frame)
                handleGame(frame)
                if exitcode == NOCAM:
                    print('Unable to detect a camera')
                    break
                renderList.append(gamePack)
                renderList.append(homeButton)
        else:
            renderList.append(pausePack)

        # Flatten all packs/objects into one list, then render sorted by z
        flatRenderList = []
        for root in renderList:
            collectRenderObjects(root, flatRenderList)

        for ob in sorted(flatRenderList, key=lambda x: x.z):
            ob.render(frame)
        cv.imshow(windowName, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        key = cv.waitKey(1) & 0xFF
        handleKeyRes = handleKey(key)
        if handleKeyRes == QUIT:
            break

    cv.destroyAllWindows()
from ultralytics import YOLO
import cv2
import pandas as pd
import pickle

class BallTracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath)

    def interpolateBallPositions(self, ballPositions):
        ballPositions = [x.get(1, []) for x in ballPositions]
        dfBallPositions = pd.DataFrame(ballPositions, columns=['x1', 'y1', 'x2', 'y2'])
        dfBallPositions = dfBallPositions.interpolate()
        dfBallPositions = dfBallPositions.bfill()
        ballPositions = [{1:x} for x in dfBallPositions.to_numpy().tolist()]

        return ballPositions

    def detectFrames(self, frames, readFromStub=False, stubPath=None):
        ballDetections = []

        if readFromStub and stubPath is not None:
            with open(stubPath, 'rb') as f:
                ballDetections = pickle.load(f)
                
            return ballDetections

        for frame in frames:
            ballDict = self.detectFrame(frame)
            ballDetections.append(ballDict)

        if stubPath is not None:
            with open(stubPath, 'wb') as f:
                pickle.dump(ballDetections, f)
        
        return ballDetections

    def detectFrame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        ballDict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ballDict[1] = result

        return ballDict
    
    def drawBoxes(self, videoFrames, ballDetections):
        outputVideoFrames = []
        for frame, ballDict in zip(videoFrames, ballDetections):
            for trackId, bbox in ballDict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {trackId}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            outputVideoFrames.append(frame)

        return outputVideoFrames
    
    def getBallShotFrames(self, ballPositions):
        ballPositions = [x.get(1, []) for x in ballPositions]
        dfBallPositions = pd.DataFrame(ballPositions, columns=['x1', 'y1', 'x2', 'y2'])
        dfBallPositions['ball_hit'] = 0
        dfBallPositions['mid_y'] = (dfBallPositions['y1'] + dfBallPositions['y2'])/2
        dfBallPositions['mid_y_rolling_mean'] = dfBallPositions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        dfBallPositions['delta_y'] = dfBallPositions['mid_y_rolling_mean'].diff()
        minimumChangeFramesForHit = 25
        for i in range(1, len(dfBallPositions) - int(minimumChangeFramesForHit * 1.2)):
            negativePositionChange = dfBallPositions['delta_y'].iloc[i] > 0 and dfBallPositions['delta_y'].iloc[i + 1] < 0
            positivePositionChange = dfBallPositions['delta_y'].iloc[i] < 0 and dfBallPositions['delta_y'].iloc[i + 1] > 0

            if negativePositionChange or positivePositionChange:
                changeCount = 0
                for changeFrame in range(i + 1, i + int(minimumChangeFramesForHit * 1.2) + 1):
                    negativePositionChangeFollowingFrame = dfBallPositions['delta_y'].iloc[i] > 0 and dfBallPositions['delta_y'].iloc[changeFrame] < 0
                    positivePositionChangeFollowingFrame = dfBallPositions['delta_y'].iloc[i] < 0 and dfBallPositions['delta_y'].iloc[changeFrame] > 0

                    if negativePositionChange and negativePositionChangeFollowingFrame:
                        changeCount += 1
                    elif positivePositionChange and positivePositionChangeFollowingFrame:
                        changeCount += 1
            
                if changeCount > minimumChangeFramesForHit - 1:
                    dfBallPositions['ball_hit'].iloc[i] = 1

        frameNumsWithBallHits = dfBallPositions[dfBallPositions['ball_hit'] == 1].index.tolist()

        return frameNumsWithBallHits
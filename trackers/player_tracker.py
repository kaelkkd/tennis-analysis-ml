from ultralytics import YOLO
from utils import measureDistance, getCenterOfBbox
import sys
sys.path.append('../')
import cv2
import pickle


class PlayerTracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath)

    def detectFrames(self, frames, readFromStub=False, stubPath=None):
        playerDetections = []

        if readFromStub and stubPath is not None:
            with open(stubPath, 'rb') as f:
                playerDetections = pickle.load(f)
                
            return playerDetections

        for frame in frames:
            playerDict = self.detectFrame(frame)
            playerDetections.append(playerDict)

        if stubPath is not None:
            with open(stubPath, 'wb') as f:
                pickle.dump(playerDetections, f)
        
        return playerDetections
    
    def chooseAndFilterPlayers(self, courtKeypoints, playerDetections):
        playerDetectionFirstFrame = playerDetections[0]
        chosenPlayer = self.choosePlayers(courtKeypoints, playerDetectionFirstFrame)
        filteredPlayerDetections = []
        for playerDict in playerDetections:
            filteredPlayerDict = {trackId: bbox for trackId, bbox in playerDict.items() if trackId in chosenPlayer}
            filteredPlayerDetections.append(filteredPlayerDict)

        return filteredPlayerDetections
    
    def choosePlayers(self, courtKeypoints, playerDict):
        distances = []
        for trackId, bbox in playerDict.items():
            playerCenter = getCenterOfBbox(bbox)
            minDistance = float('inf')
            for i in range(0, len(courtKeypoints), 2):
                courtKeypoint = (courtKeypoints[i], courtKeypoints[i+1])
                distance = measureDistance(playerCenter, courtKeypoint)
                if distance < minDistance:
                    minDistance = distance

            distances.append((trackId, minDistance))
        
        distances.sort(key = lambda x: x[1])
        chosenPlayers = [distances[0][0], distances[1][0]]

        return chosenPlayers

    def detectFrame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        idNameDict = results.names
        playerDict = {}

        for box in results.boxes:
            trackId = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            objectClsId = box.cls.tolist()[0]
            objectClsName = idNameDict[objectClsId]
            if objectClsName == "person":
                playerDict[trackId] = result

        return playerDict
    
    def drawBoxes(self, videoFrames, playerDetections):
        outputVideoFrames = []
        for frame, playerDict in zip(videoFrames, playerDetections):
            for trackId, bbox in playerDict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {trackId}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            outputVideoFrames.append(frame)

        return outputVideoFrames
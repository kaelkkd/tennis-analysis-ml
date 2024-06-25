import numpy as np
import constants
import cv2
import sys
from utils import (convertPixelDistToMeters, convertMetersToPixelDist, getFootPosition, 
                   getClosestKeypointIndex, getBboxHeight, measureXYdistance, getMiddleOfBbox)
sys.path.append('../')

class MiniCourt:
    def __init__(self, frame):
        self.drawingRectangleWidth = 250
        self.drawingRectangleHeight = 500
        self.buffer = 50
        self.courtPadding = 20
        self.setCanvasBackgroundBoxPosition(frame)
        self.setMiniCourtPosition()
        self.setCourtDrawingKeypoints()

    def convertMetersToPixels(self, meters):
        return convertMetersToPixelDist(meters, constants.DOUBLE_LINE_WIDTH, self.courtDrawingWidth)

    def setMiniCourtPosition(self):
        self.courtStartX = self.startX + self.courtPadding
        self.courtStartY = self.startY + self.courtPadding
        self.courtEndX = self.endX - self.courtPadding
        self.courtEndY =  self.endY - self.courtPadding
        self.courtDrawingWidth = self.courtEndX - self.courtStartX

    def setCanvasBackgroundBoxPosition(self, frame):
        frame = frame.copy()
        self.endX = frame.shape[1] - self.buffer
        self.endY = self.buffer + self.drawingRectangleHeight
        self.startX = self.endX - self.drawingRectangleWidth
        self.startY = self.endY - self.drawingRectangleHeight
        self.setCourtLines()

    def setCourtDrawingKeypoints(self):
        drawingKeypoints = [0] * 28
        drawingKeypoints[0] , drawingKeypoints[1] = int(self.courtStartX), int(self.courtStartY)
        # point 1
        drawingKeypoints[2] , drawingKeypoints[3] = int(self.courtEndX), int(self.courtStartY)
        # point 2
        drawingKeypoints[4] = int(self.courtStartX)
        drawingKeypoints[5] = self.courtStartY + self.convertMetersToPixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawingKeypoints[6] = drawingKeypoints[0] + self.courtDrawingWidth
        drawingKeypoints[7] = drawingKeypoints[5] 
        # #point 4
        drawingKeypoints[8] = drawingKeypoints[0] +  self.convertMetersToPixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawingKeypoints[9] = drawingKeypoints[1] 
        # #point 5
        drawingKeypoints[10] = drawingKeypoints[4] + self.convertMetersToPixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawingKeypoints[11] = drawingKeypoints[5] 
        # #point 6
        drawingKeypoints[12] = drawingKeypoints[2] - self.convertMetersToPixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawingKeypoints[13] = drawingKeypoints[3] 
        # #point 7
        drawingKeypoints[14] = drawingKeypoints[6] - self.convertMetersToPixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawingKeypoints[15] = drawingKeypoints[7] 
        # #point 8
        drawingKeypoints[16] = drawingKeypoints[8] 
        drawingKeypoints[17] = drawingKeypoints[9] + self.convertMetersToPixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawingKeypoints[18] = drawingKeypoints[16] + self.convertMetersToPixels(constants.SINGLE_LINE_WIDTH)
        drawingKeypoints[19] = drawingKeypoints[17] 
        # #point 10
        drawingKeypoints[20] = drawingKeypoints[10] 
        drawingKeypoints[21] = drawingKeypoints[11] - self.convertMetersToPixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawingKeypoints[22] = drawingKeypoints[20] +  self.convertMetersToPixels(constants.SINGLE_LINE_WIDTH)
        drawingKeypoints[23] = drawingKeypoints[21] 
        # # #point 12
        drawingKeypoints[24] = int((drawingKeypoints[16] + drawingKeypoints[18])/2)
        drawingKeypoints[25] = drawingKeypoints[17] 
        # # #point 13
        drawingKeypoints[26] = int((drawingKeypoints[20] + drawingKeypoints[22])/2)
        drawingKeypoints[27] = drawingKeypoints[21] 

        self.drawingKeypoints = drawingKeypoints
        
    def setCourtLines(self):
         self.lines = {(0, 2), (4, 5), (6, 7), (1, 3),
                       (0, 1), (8, 9), (10, 11), (2, 3)}
         
    def drawBackground(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.startX, self.startY), (self.endX, self.endY), (255, 255, 255), cv2.FILLED)
        output = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        output[mask] = cv2.addWeighted(frame, alpha, shapes, 1- alpha, 0)[mask]

        return output

    def drawCourt(self, frame):
        for i in range(0, len(self.drawingKeypoints), 2):
            x = int(self.drawingKeypoints[i])
            y = int(self.drawingKeypoints[i+1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for line in self.lines:
            startPoint = (int(self.drawingKeypoints[line[0]*2]), int(self.drawingKeypoints[line[0]*2 + 1]))
            endPoint = (int(self.drawingKeypoints[line[1]*2]), int(self.drawingKeypoints[line[1]*2 + 1]))
            cv2.line(frame, startPoint, endPoint, (0, 0, 0), 2)

        netStartPoint = (self.drawingKeypoints[0], int((self.drawingKeypoints[1] + self.drawingKeypoints[5]) / 2))
        netEndPoint = (self.drawingKeypoints[2], int((self.drawingKeypoints[1] + self.drawingKeypoints[5]) / 2))
        cv2.line(frame, netStartPoint, netEndPoint, (255, 0, 0), 2)

        return frame

    def drawMiniCourt(self, frames):
        outputFrames = []
        for frame in frames:
            frame = self.drawBackground(frame)
            frame = self.drawCourt(frame)
            outputFrames.append(frame)

        return outputFrames
    
    def getMiniCourtStartPoint(self):
        return (self.courtStartX, self.courtStartY)

    def getMiniCourtWidth(self):
        return self.courtDrawingWidth
    
    def getMiniCourtDrawingKeypoints(self):
        return self.drawingKeypoints
    
    def getMiniCourtCoordinates(self, objectPosition, closestKeyPoint, closestKeyPointIndex, playerHeightPx, playerHeightMeters):
            distanceKeyPointXPx, distanceKeyPointYPx = measureXYdistance(objectPosition, closestKeyPoint)
            distanceKeyPointXMeters = convertPixelDistToMeters(distanceKeyPointXPx, playerHeightMeters, playerHeightPx)
            distanceKeyPointYMeters = convertPixelDistToMeters(distanceKeyPointYPx, playerHeightMeters, playerHeightPx)
            
            # Convert to mini court coordinates
            miniCourtXDistPx = self.convertMetersToPixels(distanceKeyPointXMeters)
            miniCourtYDistPx = self.convertMetersToPixels(distanceKeyPointYMeters)
            
            closestMiniCourtKeyPoint = (self.drawingKeypoints[closestKeyPointIndex * 2], self.drawingKeypoints[closestKeyPointIndex * 2 + 1])
            
            miniCourtPlayerPosition = (closestMiniCourtKeyPoint[0] + miniCourtXDistPx, closestMiniCourtKeyPoint[1] + miniCourtYDistPx)
            
            return miniCourtPlayerPosition


    def convertBBToMiniCourtCoordinates(self, playerBoxes, ballBoxes, originalCourtKeyPoints):
        playerHeights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS 
        }
        outputPlayerBoxes = []
        outputBallBoxes = []

        for frameNum, playerBbox in enumerate(playerBoxes):
            ballBox = ballBoxes[frameNum][1]
            ballPosition = getMiddleOfBbox(ballBox)
            closestPlayerId = min(playerBbox.keys(), key=lambda x: measureXYdistance(ballPosition, getMiddleOfBbox(playerBbox[x])))
            outputPlayerBboxesDict = {}
            for playerId, bbox in playerBbox.items():
                footPosition = getFootPosition(bbox)
                closestKeypointIndex = getClosestKeypointIndex(footPosition, originalCourtKeyPoints, [0, 2, 12, 13])
                closestKeypoint = (originalCourtKeyPoints[closestKeypointIndex*2], originalCourtKeyPoints[closestKeypointIndex*2 + 1])

                # Get player height in pixels
                frameIndexMin = max(0, frameNum - 20)
                frameIndexMax = min(len(playerBoxes), frameNum + 50)
                bboxesHeightsPx = [getBboxHeight(playerBoxes[i][playerId]) for i in range(frameIndexMin, frameIndexMax)]
                playerMaxHeightPx = max(bboxesHeightsPx)
                miniCourtPlayerPosition = self.getMiniCourtCoordinates(
                    footPosition, closestKeypoint, closestKeypointIndex, playerMaxHeightPx, playerHeights[playerId]
                )

                outputPlayerBboxesDict[playerId] = miniCourtPlayerPosition
                if closestPlayerId == playerId:
                    closestKeypointIndex = getClosestKeypointIndex(ballPosition, originalCourtKeyPoints, [0, 2, 12, 13])
                    closestKeypoint = (originalCourtKeyPoints[closestKeypointIndex*2], originalCourtKeyPoints[closestKeypointIndex*2 + 1])
                    miniCourtBallPosition = self.getMiniCourtCoordinates(
                        ballPosition, closestKeypoint, closestKeypointIndex, playerMaxHeightPx, playerHeights[playerId]
                    )
                    outputBallBoxes.append({1: miniCourtBallPosition})
            outputPlayerBoxes.append(outputPlayerBboxesDict)

        return outputPlayerBoxes, outputBallBoxes


    def drawPointsOnMiniCourt(self, frames, positions, color=(0, 255, 0)):
        for frameNum, frame in enumerate(frames):
            for _, position in positions[frameNum].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)

        return frames
    

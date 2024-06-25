from utils import readVideo, saveVideo, measureDistance, drawPlayerStats, convertMetersToPixelDist, convertPixelDistToMeters
import constants
import cv2
import pandas as pd
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy

def main():
    #Read video
    inputVideoPath = "input_videos/input_video.mp4"
    videoFrames = readVideo(inputVideoPath)

    #Detect the players and the ball
    playerTracker = PlayerTracker(modelPath='yolov8x')
    ballTracker = BallTracker(modelPath='models/yolo5_last.pt')
    playerDetections = playerTracker.detectFrames(videoFrames, readFromStub=True, stubPath="tracker_stubs/player_detections.pkl")
    ballDetections = ballTracker.detectFrames(videoFrames, readFromStub=True, stubPath="tracker_stubs/ball_detections.pkl")
    ballDetections = ballTracker.interpolateBallPositions(ballDetections)

    #Detecting the court line
    courtModelPath = "models/keypoints_model.pth"
    courtLineDetector = CourtLineDetector(courtModelPath)
    courtKeypoints = courtLineDetector.predict(videoFrames[0])

    #Choosing the players
    playerDetections = playerTracker.chooseAndFilterPlayers(courtKeypoints, playerDetections)

    miniCourt = MiniCourt(videoFrames[0])

    #Track the ball shots
    ballShotFrames = ballTracker.getBallShotFrames(ballDetections) 

    #Convert the positions to mini court positions
    playerMiniCourtDetections, ballMiniCourtDetections = miniCourt.convertBBToMiniCourtCoordinates(playerDetections, ballDetections, courtKeypoints)

    playerStats = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ballShotIndex in range(len(ballShotFrames) - 1):
        startFrame = ballShotFrames[ballShotIndex]
        endFrame = ballShotFrames[ballShotIndex + 1]
        ballShotTimeInSeconds = (endFrame - startFrame) / 24 #For 24fps videos

        distanceCoveredBallPx = measureDistance(ballMiniCourtDetections[startFrame][1], ballMiniCourtDetections[endFrame][1])
        distanceCoveredBallMeters = convertPixelDistToMeters(distanceCoveredBallPx, constants.DOUBLE_LINE_WIDTH, miniCourt.getMiniCourtWidth())
        
        #Convert the speed to km/h
        shotSpeed = distanceCoveredBallMeters / ballShotTimeInSeconds * 3.6

        #Determines who shot the ball and its opponent
        playerPositions = playerMiniCourtDetections[startFrame]
        playerShotBall = min(playerPositions.keys(), key=lambda playerId: measureDistance(playerPositions[playerId], ballMiniCourtDetections[startFrame][1]))
        opponentPlayerId = 1 if playerShotBall == 2 else 2
        distanceCoveredOponnentPlayer = measureDistance(playerMiniCourtDetections[startFrame][opponentPlayerId], playerMiniCourtDetections[endFrame][opponentPlayerId])
        distanceCoveredOponnentPlayerMeters = convertPixelDistToMeters(distanceCoveredOponnentPlayer, constants.DOUBLE_LINE_WIDTH, miniCourt.getMiniCourtWidth())
        opponentSpeed = distanceCoveredOponnentPlayerMeters / ballShotTimeInSeconds * 3.6

        currentPlayerStats = deepcopy(playerStats[-1])
        currentPlayerStats['frame_num'] = startFrame
        currentPlayerStats[f'player_{playerShotBall}_number_of_shots'] += 1
        currentPlayerStats[f'player_{playerShotBall}_total_shot_speed'] += shotSpeed
        currentPlayerStats[f'player_{playerShotBall}_last_shot_speed'] = shotSpeed
        currentPlayerStats[f'player_{opponentPlayerId}_total_player_speed'] += opponentSpeed
        currentPlayerStats[f'player_{opponentPlayerId}_last_player_speed'] += opponentSpeed
        playerStats.append(currentPlayerStats)
    
    playerStatsDf = pd.DataFrame(playerStats)
    framesDf = pd.DataFrame({'frame_num':list(range(len(videoFrames)))})
    playerStatsDf = pd.merge(framesDf, playerStatsDf, on='frame_num', how='left')
    playerStatsDf = playerStatsDf.ffill()

    playerStatsDf['player_1_average_shot_speed'] = playerStatsDf['player_1_total_shot_speed'] / playerStatsDf['player_1_number_of_shots']
    playerStatsDf['player_2_average_shot_speed'] = playerStatsDf['player_2_total_shot_speed'] / playerStatsDf['player_2_number_of_shots']
    playerStatsDf['player_1_average_player_speed'] = playerStatsDf['player_1_total_player_speed'] / playerStatsDf['player_2_number_of_shots']
    playerStatsDf['player_2_average_player_speed'] = playerStatsDf['player_2_total_player_speed'] / playerStatsDf['player_1_number_of_shots']

    #Drawing the outputs
    #Bounding boxes
    outputVideoFrames = playerTracker.drawBoxes(videoFrames, playerDetections)
    outputVideoFrames = ballTracker.drawBoxes(outputVideoFrames, ballDetections)
    outputVideoFrames = courtLineDetector.drawKeypointsOnVideo(outputVideoFrames, courtKeypoints)

    #Mini court
    outputVideoFrames = miniCourt.drawMiniCourt(outputVideoFrames)
    outputVideoFrames = miniCourt.drawPointsOnMiniCourt(outputVideoFrames, playerMiniCourtDetections)
    outputVideoFrames = miniCourt.drawPointsOnMiniCourt(outputVideoFrames, ballMiniCourtDetections, color=(0, 255, 255))

    #Player stats
    outputVideoFrames = drawPlayerStats(outputVideoFrames, playerStatsDf)

    for i, frame in enumerate(outputVideoFrames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    saveVideo(outputVideoFrames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
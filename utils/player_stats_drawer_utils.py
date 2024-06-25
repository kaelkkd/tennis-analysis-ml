import numpy as np
import cv2

def drawPlayerStats(outputVideoFrames, playerStats):
    for index, row in playerStats.iterrows():
        player1ShotSpeed = row['player_1_last_shot_speed']
        player2ShotSpeed = row['player_2_last_shot_speed']
        player1Speed = row['player_1_last_player_speed']
        player2Speed = row['player_2_last_player_speed']

        avgPlayer1ShotSpeed = row['player_1_average_shot_speed']
        avgPlayer2ShotSpeed = row['player_2_average_shot_speed']
        avgPlayer1Speed = row['player_1_average_player_speed']
        avgPlayer2Speed = row['player_2_average_player_speed']

        frame = outputVideoFrames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 230

        startX = frame.shape[1] - 400
        startY = frame.shape[0] - 500
        endX = startX + width
        endY = startY + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (startX, startY), (endX, endY), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        outputVideoFrames[index] = frame

        text = "     Player 1     Player 2"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 80, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 10, startY + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player1ShotSpeed:.1f} km/h    {player2ShotSpeed:.1f} km/h"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 130, startY + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 10, startY + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player1Speed:.1f} km/h    {player2Speed:.1f} km/h"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 130, startY + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 10, startY + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avgPlayer1ShotSpeed:.1f} km/h    {avgPlayer2ShotSpeed:.1f} km/h"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 130, startY + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 10, startY + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avgPlayer1Speed:.1f} km/h    {avgPlayer2Speed:.1f} km/h"
        outputVideoFrames[index] = cv2.putText(outputVideoFrames[index], text, (startX + 130, startY + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return outputVideoFrames
def getCenterOfBbox(bbox):
    x1, y1, x2, y2 = bbox
    centerX = int((x1 + x2) / 2)
    centerY = int((y1 + y2) / 2)

    return (centerX, centerY)

def measureDistance(p1, p2):
    return ((p1[0] - p2[0]) **2 + (p1[1] - p2[1]) **2) **0.5

def getFootPosition(bbox):
    x1, y1, x2, y2 = bbox

    return (int((x1 + x2) / 2), y2)

def getClosestKeypointIndex(point, keypoints, keypointIndices):
    closestDistance = float('inf')
    keypointInd = keypointIndices[0]
    for keypointIndex in keypointIndices:
        keypoint = keypoints[keypointIndex*2], keypoints[keypointIndex*2 + 1]
        distance = abs(point[1] - keypoint[1])
        
        if distance < closestDistance:
            closestDistance = distance
            keypointInd = keypointIndex

    return keypointInd

def getBboxHeight(bbox):
    return bbox[3] - bbox[1]

def measureXYdistance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def getMiddleOfBbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
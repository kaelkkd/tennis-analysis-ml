def convertPixelDistToMeters(pixelDistance, referenceHeightInMeters, referenceHeightInPixels):
    return (pixelDistance * referenceHeightInMeters) / referenceHeightInPixels

def convertMetersToPixelDist(meters, referenceHeightInMeters, referenceHeightInPixels):
    return (meters * referenceHeightInPixels) / referenceHeightInMeters


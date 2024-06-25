import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, modelPath):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(modelPath, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):    
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageTensor = self.transform(imageRGB).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(imageTensor)
        keypoints = outputs.squeeze().cpu().numpy()
        originalHeight, originalWidth = image.shape[:2]
        keypoints[::2] *= originalWidth / 224.0
        keypoints[1::2] *= originalHeight / 224.0

        return keypoints

    def drawKeypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
        return image
    
    def drawKeypointsOnVideo(self, videoFrames, keypoints):
        outputVideoFrames = []
        for frame in videoFrames:
            frame = self.drawKeypoints(frame, keypoints)
            outputVideoFrames.append(frame)

        return outputVideoFrames
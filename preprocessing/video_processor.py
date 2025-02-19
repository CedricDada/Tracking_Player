import cv2
import torch
import numpy as np

class VideoProcessor:
    def __init__(self, config):
        self.roi_resolution = config['video'].get('roi_resolution', [1280, 720]) 
        self.debounce_model = torch.hub.load('AK391/animegan2-pytorch', 'generator', pretrained=True)
        
    def preprocess(self, frame):
        # Correction de la perspective
        frame = self._apply_homography(frame)
        # Débruitage léger
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        return frame
    
    def _apply_homography(self, frame):
        # Points spécifiques au terrain de football (à calibrer)
        src = np.float32([[0, 0], [frame.shape[1], 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]]])
        dst = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (1280,720))
    
    def process_roi(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h*0.1):int(h*0.9), int(w*0.2):int(w*0.8)]
        return cv2.resize(roi, tuple(self.roi_resolution))
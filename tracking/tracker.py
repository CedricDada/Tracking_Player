from deep_sort_realtime.deepsort_tracker import DeepSort

class HybridTracker:
    def __init__(self, config):
        self.deepsort = DeepSort(
            max_age=config['tracking']['max_age'],
            n_init=config['tracking']['n_init'],
            nn_budget=100,
            #suse_cuda=True  # Activer CUDA si disponible
        )
    
    def update(self, detections, frame):
        # Convertir les détections au format attendu par DeepSort
        bbs = []
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            confidence = det['confidence']
            class_id = det['class']
            embedding = det['embedding']
            
            bbs.append((bbox, confidence, class_id, embedding))
        
        # Mettre à jour le tracker
        tracks = self.deepsort.update_tracks(bbs, frame=frame)
        
        return tracks
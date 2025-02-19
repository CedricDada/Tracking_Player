import cv2

class Visualizer:
    @staticmethod
    def draw_tracks(frame, tracks):
        """
        Dessine les boîtes de détection et les IDs sur la frame.
        
        Args:
            frame (numpy.ndarray): La frame d'entrée.
            tracks (list): Liste des objets trackés.
        
        Returns:
            numpy.ndarray: La frame annotée.
        """
        for track in tracks:
            bbox = track.to_tlbr().astype(int)  # Convertir en [x1, y1, x2, y2]
            track_id = track.track_id
            
            # Dessiner la boîte de détection
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Ajouter l'ID de l'objet
            cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
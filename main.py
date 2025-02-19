import yaml
import cv2
from detection.detector import FootballDetector
from tracking.tracker import HybridTracker
from preprocessing.video_processor import VideoProcessor
from utils.visualizer import Visualizer

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def main():
    # Initialiser la figure avant la boucle principale
    plt.ion()  # Activer le mode interactif
    fig, ax = plt.subplots()  # Créer une seule figure et un axe
    
    # Charger la configuration
    with open('config/tracking_cfg.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialiser les composants
    detector = FootballDetector(config)
    tracker = HybridTracker(config)
    processor = VideoProcessor(config)
    visualizer = Visualizer()
    
    # Ouvrir la vidéo d'entrée
    cap = cv2.VideoCapture(config['video']['input_path'])
    out = cv2.VideoWriter(config['video']['output_path'], 
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                          (720, 1000))
    
    # Traitement frame par frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prétraitement de la frame
        processed = processor.preprocess(frame)
        
        # Détection des objets
        detections = detector.detect(processed, frame_count)
        
        # Tracking des objets
        tracks = tracker.update(detections, processed)
        
        # Annotation de la frame
        annotated = visualizer.draw_tracks(processed, tracks)
        
        # Sauvegarder la frame annotée
        out.write(annotated)
        
        # Convertir BGR (OpenCV) en RGB (matplotlib)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Mettre à jour l'image affichée
        if frame_count == 0:  # Pour la première frame, initialiser l'image
            im = ax.imshow(rgb_frame)
        else:
            im.set_data(rgb_frame)  # Mettre à jour les données de l'image existante
        
        # Mettre à jour le titre
        ax.set_title(f"Frame {frame_count} - Détections et tracking")
        ax.axis('off')  # Masquer les axes
        
        # Rafraîchir l'affichage
        fig.canvas.draw_idle()
        plt.pause(0.001)  # Pause nécessaire pour mettre à jour l'affichage <button class="citation-flag" data-index="10">
        
        frame_count += 1
    
    # Libérer les ressources
    cap.release()
    out.release()
    plt.ioff()  # Désactiver le mode interactif après la fin de la boucle
    plt.show()  # Afficher une dernière fois la fenêtre

if __name__ == "__main__":
    main()
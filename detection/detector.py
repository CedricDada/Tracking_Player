import yaml
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
from hydra import compose, initialize_config_dir
import tempfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class FootballDetector:
    def __init__(self, config):
        """
        Initialize the FootballDetector with necessary attributes.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.current_obj_id = 0
        self.frame_buffer = []
        self.detection_history = {}
        self.temp_dir = None

        # Initialize models
        logging.info("Initializing YOLO model...")
        self.yolo = self._init_yolo()

        logging.info("Initializing SAM2 model...")
        self.sam_predictor = self._init_sam()

        logging.info("Initializing Siamese network...")
        self.siamese = self._init_siamese()

        # Preprocess video frames
        self._preprocess_video()

    def _init_yolo(self):
        """Initialize YOLO model with ONNX optimization if configured."""
        model = YOLO(self.config["models"]["yolo"])
        if self.config["optimization"].get("use_onnx", False):
            model.export(format="onnx", task="detect")
            model = YOLO(self.config["models"]["yolo"].replace(".pt", ".onnx"))
        return model

    def _init_sam(self):
        """Initialize SAM2 video predictor."""
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()

        initialize_config_dir(
            config_dir=self.config["sam"]["config_dir"],
            version_base="1.2"
        )

        predictor = build_sam2_video_predictor(
            self.config["sam"]["model_cfg"],
            self.config["sam"]["checkpoint"],
            device=self.device
        )
        return predictor

    def _init_siamese(self):
        """Initialize Siamese network for feature extraction."""
        from .embeddings import SiameseNetwork
        model = SiameseNetwork()
        model.load_state_dict(torch.load(self.config["models"]["siamese"]))
        return model.eval().to(self.device)

    def _preprocess_video(self):
        """Preprocess video frames for SAM2."""
        video_path = self.config["video"]["input_path"]
        self.temp_dir = Path(tempfile.mkdtemp())
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Processing {frame_count} frames...")
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to SAM2 recommended resolution
            frame = cv2.resize(frame, tuple(self.config["sam"]["resize_resolution"]))
            frame_path = self.temp_dir / f"{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            self.frame_buffer.append(frame)

            if frame_idx % 100 == 0:
                logging.info(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()
        logging.info("Video preprocessing completed")

        # Initialize SAM2 state with processed frames
        logging.info("Initializing SAM2 state...")
        self.sam_state = self.sam_predictor.init_state(str(self.temp_dir))
        logging.info("SAM2 state initialized")

    def detect(self, frame, frame_idx):
        """
        Detect objects in a frame using YOLO and refine with SAM2 if needed.
        """
        results = self.yolo(frame)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                detection = {
                    "bbox": box,
                    "confidence": conf,
                    "class": cls,
                    "frame_idx": frame_idx
                }

                if conf < self.config["tracking"]["confidence_threshold"]:
                    refined_detections = self._refine_detection(detection)
                    if refined_detections:
                        detections.extend(refined_detections)
                else:
                    detection["embedding"] = self._extract_embedding(frame, box)
                    detections.append(detection)

        return detections

    def _refine_detection(self, detection):
        """
        Refine a detection using SAM2
        """
        frame_idx = detection["frame_idx"]
        box = detection["bbox"]

        # Calculate center point of the box for point prompt
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        point_prompt = np.array([[center_x, center_y]], dtype=np.float32)
        point_label = np.array([1])  # Positive prompt

        refined_detections = []

        try:
            with torch.inference_mode():
                # Get object ID or create new one
                obj_id = self._get_or_create_object_id(detection)

                # Add point prompt to SAM2 via add_new_points
                self.sam_predictor.add_new_points(
                    self.sam_state,
                    frame_idx=frame_idx,
                    points=point_prompt,
                    labels=point_label,
                    obj_id=obj_id
                )

                # Get predictions for current frame
                frame_predictions = self.sam_predictor.predict(
                    self.sam_state,
                    frame_idx=frame_idx
                )

                if frame_predictions is not None and len(frame_predictions) > 0:
                    for pred_mask in frame_predictions:
                        bbox = self._mask_to_bbox(pred_mask)
                        if bbox is not None:
                            refined_detection = detection.copy()
                            refined_detection.update({
                                "bbox": bbox,
                                "frame_idx": frame_idx,
                                "obj_id": obj_id,
                                "embedding": self._extract_embedding(
                                    self.frame_buffer[frame_idx], 
                                    bbox
                                )
                            })
                            refined_detections.append(refined_detection)

                            # Update detection history
                            self.detection_history[obj_id] = refined_detection

                # Propagate to next frames
                next_frame = min(frame_idx + self.config["sam"]["propagation_window"], len(self.frame_buffer))
                if next_frame > frame_idx:
                    for f_idx in range(frame_idx + 1, next_frame):
                        propagated_masks = self.sam_predictor.predict(
                            self.sam_state,
                            frame_idx=f_idx
                        )

                        if propagated_masks is not None and len(propagated_masks) > 0:
                            for prop_mask in propagated_masks:
                                bbox = self._mask_to_bbox(prop_mask)
                                if bbox is not None:
                                    refined_detection = detection.copy()
                                    refined_detection.update({
                                        "bbox": bbox,
                                        "frame_idx": f_idx,
                                        "obj_id": obj_id,
                                        "embedding": self._extract_embedding(
                                            self.frame_buffer[f_idx], 
                                            bbox
                                        )
                                    })
                                    refined_detections.append(refined_detection)

                                    # Update detection history
                                    self.detection_history[obj_id] = refined_detection

        except Exception as e:
            print(f"SAM2 refinement failed: {str(e)}")
            # Fallback to original detection
            detection["embedding"] = self._extract_embedding(
                self.frame_buffer[frame_idx],
                detection["bbox"]
            )
            refined_detections.append(detection)

        return refined_detections
    def _mask_to_bbox(self, mask):
        """Convert binary mask to bounding box."""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        return np.array([x1, y1, x2, y2])

    def _get_or_create_object_id(self, detection):
        """Get existing object ID or create a new one based on similarity."""
        if not self.detection_history:
            self.current_obj_id += 1
            return self.current_obj_id

        max_iou = 0
        matching_id = None

        for obj_id, prev_detection in self.detection_history.items():
            if prev_detection["frame_idx"] == detection["frame_idx"] - 1:
                iou = self._compute_iou(
                    prev_detection["bbox"],
                    detection["bbox"]
                )
                if iou > max_iou and iou > self.config["tracking"]["iou_threshold"]:
                    max_iou = iou
                    matching_id = obj_id

        if matching_id is None:
            self.current_obj_id += 1
            return self.current_obj_id

        return matching_id

    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection + 1e-6)

    def _extract_embedding(self, frame, bbox):
        """Extract embedding from image region using Siamese network."""
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(128)

        crop = cv2.resize(crop, (128, 256))
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            embedding = self.siamese(tensor.to(self.device))

        return embedding.cpu().numpy()

    def cleanup(self):
        """Clean up temporary files and resources."""
        if self.temp_dir is not None:
            import shutil
            shutil.rmtree(self.temp_dir)
        self.frame_buffer.clear()
        self.detection_history.clear()
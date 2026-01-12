import argparse
import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_yolo_segment_ultralytics

class YOLOSegment(Model):
    """YOLO segmentation model for IMX500 deployment."""

    def __init__(self, model_path: str, labels_path: str):
        """Initialize the YOLO segmentation model for IMX500 deployment."""
        super().__init__(
            model_file=model_path,
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )

        self.labels = np.genfromtxt(
            labels_path,
            dtype=str,
            delimiter="\n",
            ndmin=1
        )

    def post_process(self, output_tensors):
        """Post-process the output tensors for instance segmentation."""
        return pp_yolo_segment_ultralytics(output_tensors)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="assets/packerOut.zip")
    parser.add_argument("--labels", type=str,
                        default="assets/labels.txt")
    parser.add_argument("--threshold", type=float,
                        default=0.20)
    parser.add_argument("--fps", type=int,
                        default=5)
    parser.add_argument("--disable-draw", action="store_true")
    return parser.parse_args()

def init():
    global args, device, model, annotator, threshold, draw
    args = get_args()
    device = AiCamera(frame_rate=args.fps, image_size=(640, 480), enable_input_tensor=True)
    model = YOLOSegment(args.model, args.labels)
    device.deploy(model)
    annotator = Annotator()
    threshold = args.threshold
    draw = not args.disable_draw

def run(cb):
    global device, model, annotator, threshold
    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > threshold]
            labels = [f"{model.labels[c]}" for m, c, s, _, _ in detections]
            if (draw):
                annotator.annotate_instance_segments(frame, detections)
                annotator.annotate_boxes(frame, detections, labels=labels)
            # --- ROI crop ---
            img = frame.image
            # rx, ry, rw, rh = frame.roi  # 0.0〜1.0

            h, w, _ = img.shape
            # x1 = int(rx * w)
            # y1 = int(ry * h)
            # x2 = int((rx + rw) * w)
            # y2 = int((ry + rh) * h)
            # x1 = 0
            # y1 = 0
            # x2 = int(w / 2)
            # y2 = int(h / 2)

            # roi_img = img[y1:y2, x1:x2]

            # cb(img[0:h/2, 0:w/2])
            cb(frame.input_tensor)
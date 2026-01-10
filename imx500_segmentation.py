# Portions of this software are based on:
# https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_segmentation_demo.py

import argparse
import sys
import time
from typing import Dict

import cv2
import numpy as np

from picamera2 import MappedArray, CompletedRequest, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

def draw_mask(frame: np.ndarray, mask: np.ndarray):
    h, w = frame.shape[:2]
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_color[mask > 0] = (0, 255, 0)
    mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)
    idx = np.any(mask_color != (0, 0, 0), axis=2)
    frame[idx] = mask_color[idx]

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_deeplabv3plus.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

def init():
    global args, imx500, intrinsics, picam2

    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "segmentation"
    elif intrinsics.task != "segmentation":
        print("Network is not a segmentation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    picam2.pre_callback = None

def get_frame():
    request = picam2.capture_request()
    try:
        frame = request.make_array("main").copy()
        mask = imx500.get_outputs(metadata=request.get_metadata())[0]
        draw_mask(frame, mask)

    except Exception:
        frame = None
    finally:
        request.release()

    return frame

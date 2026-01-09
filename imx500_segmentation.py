# Portions of this software are based on:
# https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_segmentation_demo.py

import argparse
import sys
import time
from typing import Dict

import numpy as np

from picamera2 import MappedArray, CompletedRequest, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

RED = np.array([255, 0, 0, 255], dtype=np.uint8)

def create_and_draw_masks(request: CompletedRequest):
    """Create masks from the output tensor and draw them on the main output image."""
    masks = create_masks(request)
    draw_masks(request, masks)


def create_masks(request: CompletedRequest) -> Dict[int, np.ndarray]:
    """Create masks from the output tensor, scaled to the ISP output."""
    res = {}
    np_outputs = imx500.get_outputs(metadata=request.get_metadata())
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return res
    mask = np_outputs[0]
    found_indices = np.unique(mask)

    for i in found_indices:
        if i == 0:
            continue
        output_shape = [input_h, input_w, 4]
        colour = [(0, 0, 0, 0), RED.copy()]
        colour[1][3] = 150  # update the alpha value here, to save setting it later
        overlay = np.array(mask == i, dtype=np.uint8)
        overlay = np.array(colour)[overlay].reshape(output_shape).astype(np.uint8)
        # No need to resize the overlay, it will be stretched to the output window.
        res[i] = overlay
    return res


def draw_masks(request: CompletedRequest, masks: Dict[int, np.ndarray]):
    """Draw the masks directly onto the ISP output (main stream)."""
    if not masks:
        return

    # masks を合成（RGBA）
    input_w, input_h = imx500.get_input_size()
    overlay = np.zeros((input_h, input_w, 4), dtype=np.uint8)
    for v in masks.values():
        overlay += v

    # main stream に直接書き込む
    with MappedArray(request, "main") as m:
        img = m.array  # BGR, uint8

        alpha = overlay[..., 3:4] / 255.0
        fg = alpha > 0

        img[fg[..., 0]] = (
            img[fg[..., 0]] * (1.0 - alpha[fg]) +
            overlay[..., :3][fg[..., 0]] * alpha[fg]
        ).astype(np.uint8)


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
        create_and_draw_masks(request)
        frame = request.make_array("main")
    except Exception:
        frame = None
    finally:
        request.release()

    return frame

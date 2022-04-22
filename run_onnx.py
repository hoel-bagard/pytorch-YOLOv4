from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from tool.utils import load_class_names, plot_boxes_cv2, post_processing


def main():
    parser = ArgumentParser(description="Script to run the onnx model.")
    parser.add_argument("model_path", type=Path, help="Path to the onnx file.")
    parser.add_argument("image_path", type=Path, help="Path to the image to use.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    img_path: Path = args.image_path

    session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    print(f"The model expects input shape: {session.get_inputs()[0].shape}")

    img = cv2.imread(str(img_path))
    img_height = session.get_inputs()[0].shape[2]
    img_width = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # Compute
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)
    print(boxes)

    # class_names = load_class_names(namesfile)
    class_names = [str(i) for i in range(81)]
    plot_boxes_cv2(img, boxes[0], savename="predictions_onnx.jpg", class_names=class_names)


if __name__ == "__main__":
    main()

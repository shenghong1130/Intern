#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PYNQ FPGA classifier server with API-ready flower names.

This is a .py version of reference_code/server.ipynb. Run it on the PYNQ side,
not on the TonyPi Raspberry Pi.
"""

import os
import shutil
import threading

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from pynq import Overlay, allocate


CLASS_NAMES = [
    "白莲花",
    "雏菊",
    "荷花",
    "菊花",
    "腊梅",
    "兰花",
    "玫瑰花",
    "水仙花",
    "桃花",
    "樱花",
    "鸢尾花",
    "紫荆花",
]

API_FLOWER_NAMES = [
    "bailianhua",
    "chuju",
    "hehua",
    "juhua",
    "lamei",
    "lanhua",
    "meiguihua",
    "shuixianhua",
    "taohua",
    "yinghua",
    "yuanweihua",
    "zijinghua",
]


if not os.path.exists("design_1_wrapper.hwh") and os.path.exists("design_1.hwh"):
    shutil.copy("design_1.hwh", "design_1_wrapper.hwh")

print("Loading Overlay and DMA...")
overlay = Overlay("design_1_wrapper.bit")
x_buf = allocate(shape=(3, 28, 28), dtype=np.float32)
y_buf = allocate(shape=(12,), dtype=np.float32)
input_ch = overlay.axi_dma_0.sendchannel
output_ch = overlay.axi_dma_0.recvchannel
dma_lock = threading.Lock()
print("Overlay ready.")


def my_transform(img):
    img = img.resize((28, 28))
    im_data = np.array(img).astype(np.float32)
    im_data = im_data.transpose(2, 0, 1)
    for i in range(im_data.shape[0]):
        std = np.std(im_data[i, :, :])
        if std > 0:
            im_data[i, :, :] = (im_data[i, :, :] - np.mean(im_data[i, :, :])) / std
        else:
            im_data[i, :, :] = 0
    return im_data


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    img_bytes = request.files["image"].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im_data = my_transform(img_pil)

    with dma_lock:
        np.copyto(x_buf, im_data)
        if hasattr(x_buf, "flush"):
            x_buf.flush()
        output_ch.transfer(y_buf)
        input_ch.transfer(x_buf)
        input_ch.wait()
        output_ch.wait()
        if hasattr(y_buf, "invalidate"):
            y_buf.invalidate()
        best_class_idx = int(np.argmax(y_buf))
        best_confidence = float(y_buf[best_class_idx])

    flower_cn = CLASS_NAMES[best_class_idx]
    flower_api = API_FLOWER_NAMES[best_class_idx]
    print("prediction: {}/{} conf={:.4f}".format(flower_cn, flower_api, best_confidence))
    return jsonify(
        {
            "status": "success",
            "predicted_class": flower_api,
            "flower": flower_api,
            "flower_api": flower_api,
            "flower_cn": flower_cn,
            "raw_class": flower_cn,
            "class_index": best_class_idx,
            "confidence": best_confidence,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=False)

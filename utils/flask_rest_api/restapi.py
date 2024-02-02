# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import io

from flask import Flask, request
from PIL import Image

from models.common import DetectMultiBackend

app = Flask(__name__)
models = {}

DETECTION_URL = '/v1/object-detection/<model>'


@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        results = models[model](im, size=640)
        print(results)
        return 'ok'
        # if model in models:
        #     results = models[model](im, size=640)  # reduce size=320 for faster inference
        #     print(results)
        #     return results.pandas().xyxy[0].to_json(orient='records')


if __name__ == '__main__':
    models['traffic'] = DetectMultiBackend('./best.pt')
    print(models['traffic'])
    app.run(host='0.0.0.0', port=6000)  # debug=True causes Restarting with stat
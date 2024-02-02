"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:6000/v1/object-detection/traffic"
TEST_IMAGE = "example_image.png"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)

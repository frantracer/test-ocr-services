from .ocr_api import ocr_api
import boto3
import cv2
import json
import numpy as np
import traceback

class aws_ocr_api(ocr_api):
    def __init__(self, key_file_path):
        with open(key_file_path, "r") as file:
            lines = [ line.rstrip('\n') for line in file ]
            self.access_id = lines[0]
            self.acces_key = lines[1]
    
        self.client = boto3.client('rekognition', 'us-west-2',
                                   aws_access_key_id = self.access_id,
                                   aws_secret_access_key = self.acces_key)

    def process_image_url(self, image_url, result_json_path):
        raise NotImplementedError

    def process_image_file(self, image_path, result_json_path):
        with open(image_path, 'rb') as image:
            response = self.client.detect_text(Image={'Bytes': image.read()})

        with open(result_json_path, "w") as file:
            file.write(json.dumps(response, indent=4, sort_keys=True))

    def gen_graphics_results(self, image_path, result_json_path, result_graph_path):
        with open(result_json_path, 'r') as json_file:
            data = json.load(json_file)
        img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_bounds = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_text = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img_height, img_width, img_channels = img_orig.shape

        annotations = [] if 'TextDetections' not in data else data['TextDetections']
        for annotation in annotations:
            if annotation['Type'] == 'WORD':
                try:
                    text = annotation['DetectedText']
                    vertices = annotation['Geometry']['Polygon']
                    pt1 = [int(vertices[0]['X'] * img_width), int(vertices[0]['Y'] * img_height)]
                    pt2 = [int(vertices[1]['X'] * img_width), int(vertices[1]['Y'] * img_height)]
                    pt3 = [int(vertices[2]['X'] * img_width), int(vertices[2]['Y'] * img_height)]
                    pt4 = [int(vertices[3]['X'] * img_width), int(vertices[3]['Y'] * img_height)]
                    pts = np.array([pt1, pt2, pt3, pt4])
                    cv2.polylines(img_bounds, [pts], 1, (255,0,0), 2)
                    cv2.fillConvexPoly(img_text, pts, (255,0,0))
                    cv2.putText(img_text, text, tuple(pt4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                except Exception as e:
                    print("Error drawing box with data %s" % result_json_path)
                    print(traceback.format_exc())

        img_res = np.concatenate((np.concatenate((img_orig, img_bounds), axis=1), img_text), axis=1)
        cv2.imwrite(result_graph_path, img_res)
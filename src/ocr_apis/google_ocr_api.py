from .ocr_api import ocr_api
import http.client, urllib.request, urllib.parse, urllib.error, base64, io
import cv2
import json
import numpy as np
import traceback

class google_ocr_api(ocr_api):
    def __init__(self, key_file_path):
        with open(key_file_path, "r") as file:
            self.api_key = file.read().rstrip('\n')

    def process_image_url(self, image_url, result_json_path):
        headers = {
            'Content-Type': 'application/json',
        }

        json_body = {
            "requests": [
                {
                    "image": {
                        "source": {
                            "imageUri": image_url
                        }
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 5
                        }
                    ]
                }
            ]
        }

        try:
            conn = http.client.HTTPSConnection('vision.googleapis.com')
            conn.request("POST", "/v1/images:annotate?key=%s" % self.api_key, str(json_body), headers)
            
            response = conn.getresponse()
            data = response.read()
            with open(result_json_path, "w") as file:
                file.write(data.decode("UTF-8"))
                
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))

    def process_image_file(self, image_path, result_json_path):

        with io.open(image_path, 'rb') as file:
            content = base64.b64encode(file.read()).decode('UTF-8')

        headers = {
            'Content-Type': 'application/json',
        }

        json_body = {
            "requests": [
                {
                    "image": {
                        "content": content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 5
                        }
                    ]
                }
            ]
        }
        
        try:
            conn = http.client.HTTPSConnection('vision.googleapis.com')
            conn.request("POST", "/v1/images:annotate?key=%s" % self.api_key, str(json_body), headers)
            
            response = conn.getresponse()
            data = response.read()
            with open(result_json_path, "w") as file:
                file.write(data.decode("UTF-8"))

            conn.close()
        except Exception as e:
            print("Exception %s" % str(e))

    def gen_graphics_results(self, image_path, result_json_path, result_graph_path):
        with open(result_json_path, 'r') as json_file:
            data = json.load(json_file)
        img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_bounds = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_text = cv2.imread(image_path, cv2.IMREAD_COLOR)

        annotations = [] if 'textAnnotations' not in data['responses'][0] else data['responses'][0]['textAnnotations'][1:]
        for annotation in annotations:
            try:
                text = annotation['description']
                vertices = annotation['boundingPoly']['vertices']
                pt1 = [vertices[0]['x'], vertices[0]['y']]
                pt2 = [vertices[1]['x'], vertices[1]['y']]
                pt3 = [vertices[2]['x'], vertices[2]['y']]
                pt4 = [vertices[3]['x'], vertices[3]['y']]
                pts = np.array([pt1, pt2, pt3, pt4])
                cv2.polylines(img_bounds, [pts], 1, (255,0,0), 2)
                cv2.fillConvexPoly(img_text, pts, (255,0,0))
                cv2.putText(img_text, text, tuple(pt4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            except Exception as e:
                print("Error drawing box with data %s" % result_json_path)
                print(traceback.format_exc())

        img_res = np.concatenate((np.concatenate((img_orig, img_bounds), axis=1), img_text), axis=1)
        cv2.imwrite(result_graph_path, img_res)
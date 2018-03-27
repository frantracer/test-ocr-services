from .ocr_api import ocr_api
import http.client, urllib.request, urllib.parse, urllib.error, base64, io
import cv2
import json
import numpy as np
import traceback
import math

class azure_ocr_api(ocr_api):
    def __init__(self, key_file_path):
        with open(key_file_path, "r") as file:
            self.api_key = file.read().rstrip('\n')

    def process_image_url(self, image_url, result_json_path):
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.api_key
        }

        params = urllib.parse.urlencode({
            'language': 'unk',
            'detectOrientation ': 'true'
        })

        json_body = {
            "url" : image_url
        }

        try:
            conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
            conn.request("POST", "/vision/v1.0/ocr?%s" % params, str(json_body), headers)

            response = conn.getresponse()
            data = response.read()
            result_json = json.dumps(json.loads(data.decode("UTF-8")), indent=4, sort_keys=True)
            with open(result_json_path, "w") as file:
                file.write(result_json)
                
            conn.close()
        except Exception as e:
            print("Exception %s" % str(e))

    def process_image_file(self, image_path, result_json_path):

        with io.open(image_path, 'rb') as file:
            content = file.read()
    
        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self.api_key
        }

        params = urllib.parse.urlencode({
            'language': 'unk',
            'detectOrientation ': 'true'
        })

        try:
            conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
            conn.request("POST", "/vision/v1.0/ocr?%s" % params, content, headers)

            response = conn.getresponse()
            data = response.read()
            result_json = json.dumps(json.loads(data.decode("UTF-8")), indent=4, sort_keys=True)
            with open(result_json_path, "w") as file:
                file.write(result_json)

            conn.close()
        except Exception as e:
            print("Exception %s" % str(e))

    def gen_graphics_results(self, image_path, result_json_path, result_graph_path):
        with open(result_json_path, 'r') as json_file:
            data = json.load(json_file)

        img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_bounds = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_text = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img_height, img_width, img_channels = img_orig.shape

        angle = -data['textAngle']

        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        prot = np.array([int(img_width/2), int(img_height/2)])

        for region in data['regions']:
            for line in region['lines']:
                for word in line['words']:
                    try:
                        text = word['text']
                        xleft, ytop, bwidth, bheight = [ int(value) for value in word['boundingBox'].split(',') ]
                        pt1 = np.add(np.matmul(rot_matrix, np.subtract(np.array([xleft, ytop]), prot)), prot)
                        pt1 = pt1.astype(int)
                        pt2 = np.add(np.matmul(rot_matrix, np.subtract(np.array([xleft + bwidth, ytop]), prot)), prot)
                        pt2 = pt2.astype(int)
                        pt3 = np.add(np.matmul(rot_matrix, np.subtract(np.array([xleft + bwidth, ytop + bheight]), prot)), prot)
                        pt3 = pt3.astype(int)
                        pt4 = np.add(np.matmul(rot_matrix, np.subtract(np.array([xleft, ytop + bheight]), prot)), prot)
                        pt4 = pt4.astype(int)
                        pts = np.array([pt1, pt2, pt3, pt4])
                        cv2.polylines(img_bounds, [pts], 1, (255,0,0), 2)
                        cv2.fillConvexPoly(img_text, pts, (255,0,0))
                        cv2.putText(img_text, text, tuple(pt4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    except Exception as e:
                        print("Error drawing box with data %s" % result_json_path)
                        print(traceback.format_exc())

        img_res = np.concatenate((np.concatenate((img_orig, img_bounds), axis=1), img_text), axis=1)
        cv2.imwrite(result_graph_path, img_res)

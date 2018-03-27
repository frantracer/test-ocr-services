from .ocr_api import ocr_api
import cv2
import json
import numpy as np
import traceback

class watson_ocr_api(ocr_api):
    def __init__(self, key_file_path):
        pass

    def gen_graphics_results(self, image_path, result_json_path, result_graph_path):
        with open(result_json_path, 'r') as json_file:
            data = json.load(json_file)
        img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_bounds = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_text = cv2.imread(image_path, cv2.IMREAD_COLOR)

        for word in data['words']:
            try:
                text = word['word']
                xleft = word['location']['left']
                ytop = word['location']['top']
                bwidth = word['location']['width']
                bheight = word['location']['height']

                pt1 = [xleft, ytop]
                pt2 = [xleft + bwidth, ytop]
                pt3 = [xleft + bwidth, ytop + bheight]
                pt4 = [xleft, ytop + bheight]
                pts = np.array([pt1, pt2, pt3, pt4])
                cv2.polylines(img_bounds, [pts], 1, (255,0,0), 2)
                cv2.fillConvexPoly(img_text, pts, (255,0,0))
                cv2.putText(img_text, text, tuple(pt4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            except Exception as e:
                print("Error drawing box with data %s" % result_json_path)
                print(traceback.format_exc())

        img_res = np.concatenate((np.concatenate((img_orig, img_bounds), axis=1), img_text), axis=1)
        cv2.imwrite(result_graph_path, img_res)

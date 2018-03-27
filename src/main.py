from os import listdir, path
import time

from ocr_apis.azure_ocr_api import *
from ocr_apis.google_ocr_api import *
from ocr_apis.aws_ocr_api import *
from ocr_apis.watson_ocr_api import *

def main():

    api_name = 'azure' # aws google azure watson

    images_dir = "../images/"
    image_list = [images_dir + filename for filename in listdir(images_dir)]

    call_service = False
    gen_graphics_results = True

    # General configuration
    wait_secs = 0

    # Custom configuration for each service
    if api_name == 'google':
        api = google_ocr_api("../credentials/google-api.key")
        result_dir = "../results/google/"
    elif api_name == 'azure':
        api = azure_ocr_api("../credentials/azure-api.key")
        result_dir = "../results/azure/"
        wait_secs = 10
    elif api_name == 'aws':
        api = aws_ocr_api("../credentials/aws-api.key")
        result_dir = "../results/aws/"
    elif api_name == 'watson':
        api = watson_ocr_api("")
        result_dir = "../results/watson/"
    else:
        raise Exception("Invalid API name")

    for filepath in sorted(image_list):
        filename = path.basename(filepath)
        print("Processing %s ..." % filename)
        basename = path.splitext(filename)[0]
        jsonpath = result_dir + basename + ".json"
        graphpath = result_dir + basename + "-results.jpg"
        if call_service:
            time.sleep(wait_secs)
            api.process_image_file(filepath, jsonpath)
        if gen_graphics_results:
            api.gen_graphics_results(filepath, jsonpath, graphpath)

if __name__ == "__main__":
    main()

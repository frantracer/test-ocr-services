from os import path
import time
import argparse

from ocr_apis.azure_ocr_api import *
from ocr_apis.google_ocr_api import *
from ocr_apis.aws_ocr_api import *
from ocr_apis.watson_ocr_api import *

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Call an API Text Recognition API services')
    parser.add_argument('img', nargs='+',
                        help='Path to the image')
    parser.add_argument('--api', dest='api', required=True, choices=['aws','google','azure','watson'],
                        help='Path to the image')
    parser.add_argument('--gen-json', dest='gen_json', action='store_true',
                        help='Call API service to obtain JSON results')
    parser.add_argument('--gen-img', dest='gen_img', action='store_true',
                        help='Generate image with JSON results over it')
    args = parser.parse_args()

    # Parsed arguments
    api_name = args.api
    image_list = args.img
    call_service = args.gen_json
    gen_graphics_results = args.gen_img

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

    # Process each image
    for filepath in sorted(image_list):
        filename = path.basename(filepath)
        print("Processing %s ..." % filename)
        basename = path.splitext(filename)[0]
        jsonpath = result_dir + basename + ".json"
        graphpath = result_dir + basename + "-results.jpg"
        if call_service:
            print("\tCalling API ...", end='', flush=True)
            time.sleep(wait_secs)
            api.process_image_file(filepath, jsonpath)
            print(" DONE")
        if gen_graphics_results:
            print("\tGenerating image with API results ...", end='', flush=True)
            api.gen_graphics_results(filepath, jsonpath, graphpath)
            print(" DONE")

if __name__ == "__main__":
    main()

class ocr_api():
    def __init__(self, key_file_path):
        raise NotImplementedError

    def process_image_url(self, image_url, result_json_path):
        raise NotImplementedError

    def process_image_file(self, image_path, result_json_path):
        raise NotImplementedError
    
    def gen_graphics_results(self, image_path, result_json_path, result_graph_path):
        raise NotImplementedError
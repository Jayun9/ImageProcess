import json


class JsonProcess:
    def __init__(self):
        self.save_json = {
            'images': [],
            'categories': [],
            'annotations': []
        }

    def set_categories(self, categories):
        self.save_json['categories'] = categories

    def set_images(self, images):
        self.save_json['images'] = images

    def set_annotations(self, annotations):
        self.save_json['annotations'] = annotations

    def annotation_process(self, annotations_fetch):
        annotations = {}
        for annotation in annotations_fetch:
            annotations.setdefault(annotation[1], {
                "area": annotation[2],
                "bbox": json.loads(annotation[3]),
                "category_id": annotation[4],
                "id": annotation[1],
                "image_id": annotation[5],
                "isbbox": annotation[6].booleanValue(),
                "iscrowd": annotation[7].booleanValue(),
                "segmentation": []
            })
            annotations[annotation[1]]["segmentation"].append(json.loads(annotation[8]))
        annotation = [anno for anno in annotations.values()]
        return annotation

    def categories_process(self, categories_fetch):
        categories = [{"id": category[1],
                       "name": category[3],
                       "supercategory": category[4],
                       "color": category[2]} for category in categories_fetch]
        return categories

    def images_process(self, images_fetch):
        file_name, height, image_id, path, width = images_fetch[2:]
        image = {
            "file_name": file_name,
            "height": height,
            "id": image_id,
            "path": path,
            "width": width
        }
        return image

    def json_save(self, save_path):
        with open(save_path, 'w') as save_js_file:
            json.dump(self.save_json, save_js_file)

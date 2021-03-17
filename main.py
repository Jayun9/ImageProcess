import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def main(name):
    image_read_path = "data/16bit png image"
    json_read_path = "data/16bit png json"
    image_save_path = "image"
    json_save_path = "json"
    json_file_list = os.listdir(json_read_path)
    for json_file in json_file_list:
        save_json = {
            'images': None,
            'categories': None,
            'annotations': None
        }
        annotation_list = []
        image_list = []
        with open(os.path.join(json_read_path, json_file)) as js_file:
            json_data = json.load(js_file)
        image_dict, annotation_dict, categories_dict = data_setting(json_data)
        save_json['categories'] = categories_dict
        for image_id, image in image_dict.items():
            # 이미지 읽어오기
            image_width, image_height = image['width'], image['height']
            src_file = image['file_name']
            src = cv.imread(os.path.join(image_read_path, src_file), cv.IMREAD_UNCHANGED)
            if src is None:
                print(src_file, "파일 없음")
                continue
            src_height, src_width = src.shape[0], src.shape[1]
            annotations = annotation_dict[image_id]

            # 사이즈 변환되어 있는 이미지가 있을때 처리
            image.update({'width': 1000})
            image.update({'height': 1156})
            resize_annotation = get_resize_annotation(src_height, src_width, annotations, image_width, image_height)
            annotations = resize_annotation

            # 배경 이미지 합성
            bbox_list = [annotation['bbox'] for annotation in annotations]
            outer_most_bbox = get_outer_most_bbox(bbox_list)
            result_img, trans_w, trans_h = manipulation_background(outer_most_bbox, src, 1000, 1156, src_file)
            new_annotations = get_manipulation_annotation(annotations, trans_w, trans_h)
            for new_annotation in new_annotations:
                annotation_list.append(new_annotation)
            image.update({'width': result_img.shape[1]})
            image.update({'height': result_img.shape[0]})
            image_list.append(image)
            cv.imwrite(os.path.join(image_save_path, image['file_name']), result_img)
        save_json['images'] = image_list
        save_json['annotations'] = annotation_list
        with open(os.path.join(json_save_path, json_file), 'w') as save_js_file:
            json.dump(save_json, save_js_file)


# 이미지 경로를 얻기 위해서
def get_file_folder(file_name, image_read_path):
    image_path = "_".join(os.path.splitext(file_name)[0].split("_")[:-1])
    return os.path.join(image_read_path, image_path, file_name)


# 합성 이미지에 어노테이션을 반환
def get_manipulation_annotation(annotations, trans_w, trans_h):
    for annotation in annotations:
        new_bbox = get_manipulation_bbox(annotation['bbox'], trans_w, trans_h)
        new_segmentation = get_manipulation_segmentation(annotation['segmentation'], trans_w, trans_h)
        annotation.update({'bbox': new_bbox})
        annotation.update({'segmentation': new_segmentation})
    return annotations


# 합성 이미지에 segmentation 을 구함
def get_manipulation_segmentation(segmentation, trans_w, trans_h):
    segment_list = []
    for segment in segmentation:
        segment_array = np.array(segment)
        new_seg = np.zeros(segment_array.shape)
        new_seg[::2] = segment_array[::2] + trans_w
        new_seg[1::2] = segment_array[1::2] + trans_h
        new_seg = np.round(new_seg, 2)
        segment_list.append(new_seg.tolist())
    return segment_list


# 합성 이미지에 새로운 bbox 를 구함
def get_manipulation_bbox(bbox, trans_w, trans_h):
    new_bbox = [bbox[0] + trans_w, bbox[1] + trans_h, bbox[2], bbox[3]]
    return new_bbox


# w: h: 배경에 이미지 합성
def manipulation_background(bbox, src, width, height, src_file):
    if len(src.shape) == 2:
        background = np.zeros((height, width), dtype=np.uint16) + 65534
        is_gray = True
    else:
        background = np.zeros((height, width, 3), dtype=np.uint16) + 65534
    back_center_w = int(width / 2)
    back_center_h = int(height / 2)
    x = int(bbox['x'])
    y = int(bbox['y'])
    w = int(bbox['w'])
    h = int(bbox['h'])
    roi_w = back_center_w - int(w / 2)
    roi_h = back_center_h - int(h / 2)
    trans_w = roi_w - x
    trans_h = roi_h - y
    try:
        if is_gray:
            background[roi_h: roi_h + h, roi_w: roi_w + w] = src[y: y + h, x: x + w]
        else:
            background[roi_h: roi_h + h, roi_w: roi_w + w, :] = src[y: y + h, x: x + w, :]
    except Exception as e:
        print(e)
        print("src_shape : ", src.shape)
        print("src_w : ", w)
        print("src_h : ", h)
        print("src_name : ", src_file)
    return background, trans_w, trans_h


# 사이즈 변환에 따른 어노테이션 정보를 변환 시킴
# bbox, segmentation, area 변환
def get_resize_annotation(src_height, src_width, annotations, width, height):
    fx = src_width / width
    fy = src_height / height
    area_ratio = (src_height * src_width) / (width * height)
    for annotation in annotations:
        new_area = annotation['area'] * area_ratio
        new_segmentation = get_resize_segmentation(annotation['segmentation'], fx, fy)
        new_bbox = get_resize_bbox(annotation['bbox'], fx, fy)
        annotation.update({'area': new_area})
        annotation.update({'segmentation': new_segmentation})
        annotation.update({'bbox': new_bbox})
    return annotations


# 사이즈 변환에 따른 bbox 정보를 변환 시킴
def get_resize_bbox(bbox, fx, fy):
    new_bbox = [round(bbox[0] * fx, 2), round(bbox[1] * fy, 2), round(bbox[2] * fx, 2), round(bbox[3] * fy, 2)]
    return new_bbox


# 사이즈 변환에 따른 segmentation 정보를 변환 시킴
def get_resize_segmentation(segmentations, fx, fy):
    segment_list = []
    for segment in segmentations:
        segment_array = np.array(segment)
        new_seg = np.zeros(segment_array.shape)
        new_seg[::2] = segment_array[::2] * fx
        new_seg[1::2] = segment_array[1::2] * fy
        new_seg = np.round(new_seg, 2)
        segment_list.append(new_seg.tolist())
    return segment_list


# 제일 바깥쪽 bbox 를 구한다. 여러 어노테이션 합치는 bbox
def get_outer_most_bbox(bbox_list):
    x = [bbox[0] for bbox in bbox_list]
    y = [bbox[1] for bbox in bbox_list]
    w = [bbox[0] + bbox[2] for bbox in bbox_list]
    h = [bbox[1] + bbox[3] for bbox in bbox_list]
    outermost_bbox = {
        'x': min(x),
        'y': min(y),
        'w': max(w) - min(x),
        'h': max(h) - min(y),
    }
    return outermost_bbox


# image id 기준 딕셔너리 생성
def data_setting(json_data):
    annotation_key_image_id = {}
    image_key_image_id = {}
    for annotation in json_data['annotations']:
        annotation_key_image_id[annotation['image_id']] = []
    for annotation in json_data['annotations']:
        annotation_key_image_id[annotation['image_id']].append(annotation)
    for image in json_data['images']:
        image_key_image_id[image['id']] = image
    return image_key_image_id, annotation_key_image_id, json_data['categories']


# 확인용
def confirm(annotations, src):
    for annotation in annotations:
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(src, pt1=pt1, pt2=pt2, color=0, thickness=2)
        contours = get_contours(segmentation)
        for contour in contours:
            cv.drawContours(src, [contour], 0, 255, 2)
    return src


# 확인용
def get_contours(segmentations):
    contours_list = []
    for segment in segmentations:
        seg_arr = np.array(segment, dtype=np.int32)
        contours_len = int(len(seg_arr) / 2)
        contours = np.zeros((contours_len, 1, 2))
        contours[:, 0, 0] = seg_arr[::2]
        contours[:, 0, 1] = seg_arr[1::2]
        contours = contours.astype(np.int32)
        contours_list.append(contours)
    return contours_list


if __name__ == '__main__':
    main('PyCharm')


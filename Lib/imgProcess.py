import cv2 as cv
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


class ImgProcess:
    def __init__(self):
        self.hangle = False
        self.background = None

    def korean_path(self, img_full_path):
        stream = open(img_full_path.encode("utf-8"), "rb")
        byte_array = bytearray(stream.read())
        img_array = np.asarray(byte_array, dtype=np.uint16)

        return cv.imdecode(img_array, cv.IMREAD_UNCHANGED)

    def imread(self, img_full_path):
        img = cv.imread(img_full_path, cv.IMREAD_UNCHANGED)
        if img is None:
            img_array = np.fromfile(img_full_path, np.uint16)
            img = cv.imdecode(img_array, cv.IMREAD_UNCHANGED)
            self.hangle = True
        else:
            self.hangle = False
        return img

    def korean_path_save(self, filename, img, params=None):
        try:
            ext = os.path.splitext(filename)[1]
            result, n = cv.imencode(ext, img, params)
            if result:
                with open(filename, mode='w+b') as f:
                    n.tofile(f)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False

    def img_write(self, img_full_path, img):
        if self.hangle:
            self.korean_path_save(img_full_path, img)
        else:
            cv.imwrite(img_full_path, img)

    # 나중에 주석 지울것 -> 백그라운드는 데이터 타입이 바뀌게 되면 다시 만들어 (사이즈는 제일 큰걸로 맞춰서 하니까 괜찮음)
    # 사이즈는 최대 크기에 +2 픽셀 만큼
    def make_background(self, shape, data_type):
        background = np.zeros(shape, dtype=data_type)
        if data_type == np.uint16:
            self.background = background + 65534
        elif data_type == np.uint8:
            self.background = background + 255

    # 이미지 자르지 않고 합성하는거
    # 이동해야하는 정보 반환
    def composite_background(self, src, width, height):
        dst = copy.deepcopy(self.background)
        dst_height, dst_width = dst.shape[0], dst.shape[1]
        # 합성이 시작되는 위치
        composite_width = int((dst_width - width) / 2)
        composite_height = int((dst_height - height) / 2)

        if composite_height <= 0 or composite_width <= 0:
            print()
            raise Exception("백그라운드 사이즈를 초과함.")

        trans_w = composite_width
        trans_h = composite_height

        if len(src.shape) == 3:
            dst[composite_height:src.shape[0] + composite_height, composite_width:src.shape[1] + composite_width,
            :] = src
        else:
            dst[composite_height:src.shape[0] + composite_height, composite_width:src.shape[1] + composite_width] = src
        return dst, trans_w, trans_h

    def get_new_annotation(self, annotations, trans_w, trans_h):
        for annotation in annotations:
            new_bbox = self.get_new_bbox(annotation['bbox'], trans_w, trans_h)
            new_segmentation = self.get_new_segmentation(annotation['segmentation'], trans_w, trans_h)
            annotation.update({'bbox': new_bbox})
            annotation.update({'segmentation': new_segmentation})
        return annotations

    def get_new_bbox(self, bbox, trans_w, trans_h):
        new_bbox = [bbox[0] + trans_w, bbox[1] + trans_h, bbox[2], bbox[3]]
        return new_bbox

    def get_new_segmentation(self, segmentation, trans_w, trans_h):
        segment_list = []
        for segment in segmentation:
            segment_array = np.array(segment)
            new_seg = np.zeros(segment_array.shape)
            new_seg[::2] = segment_array[::2] + trans_w
            new_seg[1::2] = segment_array[1::2] + trans_h
            new_seg = np.round(new_seg, 2)
            segment_list.append(new_seg.tolist())
        return segment_list

    # 확인용
    def confirm(self, annotations, src):
        for annotation in annotations:
            segmentation = annotation['segmentation']
            bbox = annotation['bbox']
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(src, pt1=pt1, pt2=pt2, color=(255, 255, 255), thickness=10)
            contours = self.get_contours(segmentation)
            for contour in contours:
                cv.drawContours(src, [contour], 0, (255, 255, 255), 10)
        plt.imshow(src)
        plt.show()

    # 확인용
    def get_contours(self, segmentations):
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

    def resize_img(self, src, fx, fy):
        dst = cv.resize(src, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)
        return dst, fx, fy

    def get_resize_segmentation(self, segmentations, fx, fy):
        segment_list = []
        for segment in segmentations:
            segment_array = np.array(segment)
            new_seg = np.zeros(segment_array.shape)
            new_seg[::2] = segment_array[::2] * fx
            new_seg[1::2] = segment_array[1::2] * fy
            new_seg = np.round(new_seg, 2)
            segment_list.append(new_seg.tolist())
        return segment_list

    def get_resize_bbox(self, bbox, fx, fy):
        new_bbox = [round(bbox[0] * fx, 2), round(bbox[1] * fy, 2), round(bbox[2] * fx, 2), round(bbox[3] * fy, 2)]
        return new_bbox

    def get_resize_annotation(self, annotations, fx, fy):
        for annotation in annotations:
            new_area = annotation['area'] * fx
            new_segmentation = self.get_resize_segmentation(annotation['segmentation'], fx, fy)
            new_bbox = self.get_resize_bbox(annotation['bbox'], fx, fy)
            annotation.update({'area': new_area})
            annotation.update({'segmentation': new_segmentation})
            annotation.update({'bbox': new_bbox})
        return annotations

    # 제일 바깥족에 있는 bbox로 모은다.
    def get_outer_most_bbox(self, annotations):
        bbox_list = [annotation['bbox'] for annotation in annotations]
        x = [bbox[0] for bbox in bbox_list]
        y = [bbox[1] for bbox in bbox_list]
        w = [bbox[0] + bbox[2] for bbox in bbox_list]
        h = [bbox[1] + bbox[3] for bbox in bbox_list]
        outer_most_bbox = [min(x), min(y), max(w) - min(x), max(h) - min(y)]
        return outer_most_bbox

    def set_bbox_src(self, src, annotations):
        outer_most_bbox = self.get_outer_most_bbox(annotations)
        x = int(outer_most_bbox[0])
        y = int(outer_most_bbox[1])
        w = int(outer_most_bbox[2])
        h = int(outer_most_bbox[3])
        if len(src.shape) == 2:
            dst = src[y:y + h, x:x + w]
        else:
            dst = src[y:y + h, x:x + w, :]
        trans_x = -x
        trans_y = -y
        new_annotation = self.get_new_annotation(annotations, trans_x, trans_y)
        return dst, new_annotation

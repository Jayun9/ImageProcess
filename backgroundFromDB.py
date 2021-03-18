from Lib.dbProcess import DbProcess
from Lib.imgProcess import ImgProcess
from Lib.jsonProcess import JsonProcess

import numpy as np
import os


# DB 여러개 따로 저장할 수 있을때 구현
def temp():
    # 겨로 정보 설정
    # image_base_path : 카테고리 별로 이미지 있는 최상단 폴더
    # image_read_path : 이미지가 있는 폴더/ auto 나 unsigned 폴더가 있을 수 있음
    # image_read_paths : unsigned 나 auto 폴더가 있을 경우를 대비해서 list 안에 image_read_path 집어 넣음
    image_base_path = "D:\CT_16"
    image_dirs = os.listdir(image_base_path)

    for image_dir in image_dirs:
        image_read_path = os.path.join(image_base_path, image_dir)
        is_auto_path = os.listdir(image_read_path)
        # auto_path 가 있을 경우
        if len(is_auto_path) < 3:
            image_read_paths = [os.path.join(image_read_path, path) for path in is_auto_path]
        else:
            image_read_paths = [image_read_path]
        for image_read_path in image_read_paths:
            image_save_path = os.path.join(image_read_path, "image")
            json_save_path = os.path.join(image_save_path, "json")
            if not os.path.lexists(image_save_path):
                os.makedirs(image_save_path)
                os.makedirs(json_save_path)
            main(image_read_path, image_save_path, json_save_path)


def main():
    img_process = ImgProcess()
    json_process = JsonProcess()
    # 경로 정보
    image_read_path = r"C:\Users\OpenILab\PycharmProjects\imageProcess\data\DAQ_16\image"
    image_save_path = f"{image_read_path}/image"
    json_save_path = f"{image_save_path}/json"
    if not os.path.lexists(image_save_path):
        os.makedirs(image_save_path)
        os.makedirs(json_save_path)

    # DB 정보
    db_driver = 'org.h2.Driver'
    db_address = 'jdbc:h2:tcp://localhost:9092/~/coco'
    id_password = ['sa', ""]
    jar_address = "c:/Program Files (x86)/H2/bin/h2-1.4.199.jar"

    # Db 컨넥션 얻기
    db = DbProcess(db_driver, db_address, id_password, jar_address)

    # img_query = 이미지 가져오는 쿼리
    # img_query = "select * from IMAGES limit 5"
    img_query = "select * from IMAGES"
    images = db.execute(img_query)

    # category_query = 카테고리 가져오는 쿼리
    category_query = "select * from categories"
    categories_fetch = db.execute(category_query)
    categories = json_process.categories_process(categories_fetch)
    json_process.set_categories(categories)

    # 이미지 정보
    shape = (1156, 1200)
    data_type = np.uint16
    img_process.make_background(shape, data_type)
    annotation_list = []
    image_list = []
    image_total_count = len(images)
    cur_count = 1
    version = int(input("ct 이미지 이면 1 입력: "))
    # version = 1
    batch = 1
    for image_fetch in images:
        image = json_process.images_process(image_fetch)

        # annotation_query = 이미지에 매칭되는 어노테이션을 가져옴.
        annotation_query = f"select * from annotations where annotations.imageid = {image['id']}"
        annotations_fetch = db.execute(annotation_query)
        annotation = json_process.annotation_process(annotations_fetch)

        img_full_path = f'{image_read_path}/{image["file_name"]}'
        src = img_process.imread(img_full_path)

        if src is None:
            print(image['file_name'] + "파일이 없습니다")
            continue
        # ct 이미지일 경우
        # fx와 fy의 비율은 같아야 한다.
        # 크기가 같은 ct 일때
        if version == 1:
            src, fx, fy = img_process.resize_img(src, 0.1, 0.1)
            annotation = img_process.get_resize_annotation(annotation, fx, fy)
            image['width'] = image['width'] * fx
            image['height'] = image['height'] * fy

        try:
            dst, trans_w, trans_h = img_process.composite_background(src, image['width'], image['height'])
        except:
            dst, annotation = img_process.set_bbox_src(src, annotation)
            dst, trans_w, trans_h = img_process.composite_background(dst, dst.shape[1], dst.shape[0])
        new_annotations = img_process.get_new_annotation(annotation, trans_w, trans_h)
        for new_annotation in new_annotations:
            annotation_list.append(new_annotation)
        image.update({'width': shape[1]})
        image.update({'height': shape[0]})
        image_list.append(image)
        img_process.img_write(os.path.join(image_save_path, image['file_name']).replace('\\', '/'), dst)
        # img_process.confirm(new_annotations, dst)
        if batch == 50:
            print(round((cur_count / image_total_count) * 100, 2), " %")
            batch = 1
        cur_count += 1
        batch += 1
    # json 저장
    json_process.set_images(image_list)
    json_process.set_annotations(annotation_list)
    json_process.json_save(os.path.join(json_save_path, "result.json"))
    db.close()


if __name__ == "__main__":
    main()

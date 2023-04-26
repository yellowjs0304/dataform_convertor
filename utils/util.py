import os
import yaml
import glob
import json
import xml.etree.ElementTree as ET
from os import path as Path
from PIL import Image
from jinja2 import PackageLoader, Environment
from xml.sax.saxutils import escape
import re
import numpy as np

def get_class(classes_path):
    with open(classes_path, 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def getImagesInDir(dir_path):
    # Get the Image set list from directory
    files = glob.glob(dir_path+"/*")
    file_list_image = [file for file in files if file.endswith((".png", ".tif", ".tiff", ".TIF", ".jpeg", ".jpg", ".PNG", ".JPG"))]
    return sorted(file_list_image)

def getImagesFromFile(data_path, f_path):
    # Get the Image set list from file
    try:
        f = open(f_path, "r")
    except:
        AssertionError("File is not exist in ""%s"". Please check your data."%f_path)
    img_lst = f.readlines()
    img_lst = [data_path+val.rstrip("\n") for val in img_lst]
    f.close()
    return sorted(img_lst)

    
def getGTInDir(data_path, data_type, image_paths):
    data_lst = []
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        basename_no_ext = os.path.splitext(basename)[0]
        
        # Get GT data
        if os.path.isdir(data_path):# Loading from DIR
            if data_type == "pascalvoc":#XML
                target_path = os.path.join(data_path, "%s.xml"%basename_no_ext)
            else: # YOLO, COCO
                target_path = os.path.join(data_path, "%s.txt"%basename_no_ext)
            data_lst.append(target_path)
    return sorted(data_lst)


def getGTFromFile(data_path, data_type, image_paths):
    gt_paths = []
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        if data_type == "labelme":#JSON
            target_path = os.path.splitext(img_path)[0]+".json"
        gt_paths.append(target_path)
        
    return gt_paths

def read_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    return config


def loadGT(type, in_file): 
    if type == "yolo" or type == "coco":
        with open(in_file, 'r') as f:
            items = f.readlines()
        gt_data = [c.strip() for c in items]# return list of string

    elif type == "voc":
        tree = ET.parse(in_file)
        gt_data = tree.getroot()# return ElementTree data

    else: #type == "labelme"
        with open(in_file, "r", encoding="utf-8-sig") as json_file:
            gt_data = json.load(json_file) # return dict data
    return gt_data

class VocWriter:
    """
    from
    https://github.com/EvitanRelta/polygon-pascalvoc-writer/blob/master/polygon_pascalvoc_writer/voc_writer.py
    """
    def __init__(self, imageDir, annotationDir, imageName, image_path, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('polygon_pascalvoc_writer','templates'),keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')
        
        self.imageDir = imageDir        
        self.annotationDir = annotationDir
        self.imageName = imageName
        self.image_path = image_path # Raw Image path(original image from engine.SthConvert)

        self.template_parameters = {
            'depth': depth,
            'database': escape(database),
            'segmented': segmented,
            'objects': []
        }
    
    def nextImage(self, imageName, depth=3, database='Unknown', segmented=0):
        self.imageName = imageName
        self.template_parameters = {
            'depth': depth,
            'database': escape(database),
            'segmented': segmented,
            'objects': []
        }
        

    def addBndBox(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': escape(name),
            'pose': escape(pose),
            'truncated': truncated,
            'difficult': difficult,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
            
    def addPolygon(self, name, points, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': escape(name),
            'pose': escape(pose),
            'truncated': truncated,
            'difficult': difficult,
            'points': points,
            'xmin': min(points, key=lambda x:x[0])[0],
            'ymin': min(points, key=lambda x:x[1])[1],
            'xmax': max(points, key=lambda x:x[0])[0],
            'ymax': max(points, key=lambda x:x[1])[1]
        })
        import pdb; pdb.set_trace
    
    def save(self):
        imagePath = Path.join(self.imageDir, self.imageName)
        imageWidth, imageHeight = self.getImageSize()
        _ = self.template_parameters
        _['filename'] = escape(Path.basename(self.imageName))
        _['folder'] = escape(Path.basename(Path.abspath(self.imageDir)))
        _['path'] = escape(Path.abspath(imagePath))
        _['width'] = imageWidth
        _['height'] = imageHeight
        
        annotationName = Path.splitext(self.imageName)[0] + '.xml'
        annotationPath = Path.join(self.annotationDir, annotationName)
        with open(annotationPath, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)
    
    def getImageSize(self):
        # imagePath = Path.join(self.imageDir, self.imageName)
        
        imageWidth, imageHeight = Image.open(self.image_path).size
        return imageWidth, imageHeight
    
def getCocoDict(classes, task):
    
    output_json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    # Select Category Dict
    if task == "detect" or task == "segment":
        category_dict = {
            "id": 1,
            "name": None,
            "supercategory": "box"
        }
    elif task == "panoptic":
        category_dict = {
            "id": 1,
            "name": None,
            "supercategory": "box",
        }
    elif task == "keypoints":
        category_dict ={
            "supercategory": "box",
            "id": 1,
            "name": None,
            "keypoints": [
                # You must change this part according to your dataset
                "center",
                "left_top",
                "right_top",
                "right_bottom",
                "left_bottom"
            ],
            "skeleton":[
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5]
            ]
        }
    
    ##########
    category_dict_lst = []
    for class_nm in classes:
        category_dict["name"] = class_nm
        new_category_dict = category_dict.copy()
        category_dict_lst.append(new_category_dict)
    
    return output_json_dict, category_dict_lst

def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def recoord(polygon_coord):
    """
    4 point의 폴리곤을 받아 왼쪽 위의 좌표를 기준으로 시계방향으로 정렬해주는 함수
    
    Input:
        polygon_coord : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [4,2] shape의 array
    
    Output:
        polygon_coord : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [4,2] shape의 시계방향으로 정렬된 array
    
    """
    coord = np.array(polygon_coord)
    distance = []

    for idx, c in enumerate(coord):
        x = c[0]
        y = c[1]
        
        zero_distance = x+y
        distance.append([idx, zero_distance])

    distance = sorted(distance, key=lambda x : x[1])

    left_top = coord[distance[0][0]]
    right_bottom = coord[distance[-1][0]]

    box1 = coord[distance[1][0]]
    box2 = coord[distance[2][0]]

    if box1[0] > box2[0]:
        left_bottom = box2
        right_top = box1
    else:
        left_bottom = box1
        right_top = box2

    x1,y1 = left_top
    x2,y2 = right_top
    x3,y3 = right_bottom
    x4,y4 = left_bottom
    
    polygon = [[x1,y1],
                [x2,y2],
                [x3,y3],
                [x4,y4]]
    
    return np.array(polygon)

from shapely.geometry import Polygon, MultiPolygon
import numpy as np
from utils.util import recoord

def toVoc(args, writer, task, obj_nm, polygon):
    if task == "detect":
        writer.addPolygon(obj_nm, polygon)
    else:
        AssertionError("Pascal VOC format is only for detection task.")
        
    return writer

def toYolo(args, polygon, img_w, img_h, obj_id, task):
    
    dw, dh = 1./img_w, 1./img_h
    w = max(polygon[0::2]) - min(polygon[0::2])
    h = max(polygon[1::2]) - min(polygon[1::2])
    
    # Convert Json to Yolov8 TXT format
    if task == "detect":
        # Detection line shape = [cls_idx, center_x, center_y, w, h]
        bbox = [min(polygon[0::2]) + w/2, min(polygon[1::2]) + h/2, w, h]
        # Normalize
        bbox = [str(obj_id), bbox[0]*dw, bbox[1]*dh, bbox[2]*dw, bbox[3]*dh]
        
        new_line =" ".join([str(a) for a in bbox])+'\n'
        
    elif task == "segm":
        # Segment line shape = [cls_idx, x1, y1, x2, y2, x3, y3, x4, y4]
        # just polygon shaped bbox
        
        polygon[::2] = [point*dw for point in polygon[::2]]
        polygon[1::2] = [point*dh for point in polygon[1::2]]
        new = [str(obj_id)]
        for val in polygon:
            if val > 1.0:
                new.append(1.0)
            elif val < 0.0:
                new.append(0.0)
            else:
                new.append(val)
        
        new_line =" ".join([str(a) for a in new])+'\n'
    
    elif task == "keypoint":
        # Keypoint line shape = [cls_idx, center_x, center_y, w, h, x1, y1, v1, x2, y2, v2, x3, y3, v3, x4, y4, v4..]
        
        bbox = [min(polygon[0::2]) + w/2, min(polygon[1::2]) + h/2, w, h]
        bbox = [str(obj_id), bbox[0]*dw, bbox[1]*dh, bbox[2]*dw, bbox[3]*dh]
        
        polygon[::2] = [point*dw for point in polygon[::2]]
        polygon[1::2] = [point*dh for point in polygon[1::2]]
        new = []
        for idx, val in enumerate(polygon):
            if val > 1.0:
                new.append(1.0)
            elif val < 0.0:
                new.append(0.0)
            else:
                new.append(val)
            if idx in [1, 3, 5, 7]:
                new.append(2.0)
        
        new_line =" ".join([str(a) for a in bbox])+" "+" ".join([str(a) for a in new]) + '\n'
    else:
        raise ValueError("Unknown task type")
    return new_line

def toCoco(args, polygon, img_w, img_h, obj_nm, obj_id, task, coco_dict):
    num_keypoints = 0
    keypoints = []
    # Convert Json to Yolov8 TXT format
    if task == "detect" or "segm":
        w = max(polygon[0::2]) - min(polygon[0::2])
        h = max(polygon[1::2]) - min(polygon[1::2])
        # Detection line shape = [cls_idx, center_x, center_y, w, h]
        bbox_points = [min(polygon[0::2]) + w/2, min(polygon[1::2]) + h/2, w, h]
        category_id = obj_id
        segmentations = polygon
        poly_points =[[x, y] for x, y in zip(polygon[0::2], polygon[1::2])]
        poly_points = recoord(poly_points)
        area = MultiPolygon([Polygon(poly_points)]).area
        
    elif task == "keypoints":
        # generate keypoint
        num_keypoints = 5
        keypoints = []
        center_coord = bbox_points[0:2]
        keypoints += center_coord
        keypoints += [2]
        for points in zip(polygon[0::2], polygon[1::2]):
            x,y = points
            v = 2
            keypoints += [x,y,v]
    else:
        raise ValueError("Unknown task type")
    
    object_info = {
        "segmentation": [segmentations],
        "iscrowd": 0,
        "image_id": len(coco_dict['images']),
        "category_id": category_id,
        "id": len(coco_dict['annotations'])+1,
        "bbox": bbox_points,
        "area": area,
        "num_keypoints": num_keypoints,
        "keypoints": np.array(keypoints).tolist()
        }
    return object_info

def toDota(args, polygon, img_w, img_h, obj_nm, task):
    
    if args["normalize"]:
        dw, dh = 1./img_w, 1./img_h
        polygon[::2] = [point*dw for point in polygon[::2]]
        polygon[1::2] = [point*dh for point in polygon[1::2]]
        
        # if point > 1.0 return 1.0, if point < 0.0 return 0.0, else return point
        polygon = [1 if point > 1.0 else 0 if point < 0.0 else point for point in polygon]
    # Convert Json to Yolov8 TXT format
    if task == "detect":
        # Segment line shape = [x1, y1, x2, y2, x3, y3, x4, y4, category, difficult]
        # just polygon shaped bbox
        new = polygon
        
        difficulty = "0"
        new += [obj_nm, difficulty]# 0 -> difficult
        new_line =" ".join([str(a) for a in new])+'\n'
    else:
        raise ValueError("Dota is only supported Rotated Object Detection task.")
    return new_line

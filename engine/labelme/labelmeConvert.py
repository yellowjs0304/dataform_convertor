import os
from engine.labelme.labelme_toSth import toYolo, toCoco, toVoc, toDota
#from polygon_pascalvoc_writer import VocWriter
from utils.util import VocWriter, getCocoDict
"""
    Reference: https://github.com/EvitanRelta/polygon-pascalvoc-writer    
"""
def labelmeConvert(args, to_t, image_path, gt_data, classes, output_path, task="detection", coco_dict=None):
    # Convert Labelme Annotation to various data format
    image_name = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(image_name)[0]
    
    if to_t in ["yolo", "dota"]:
        out_file = open(output_path + '/labels/' + basename_no_ext + '.txt', 'w')
    elif to_t == "voc":
        writer = VocWriter(output_path + '/images/', output_path + '/labels/', image_name, image_path)
    elif to_t == "coco":
        image_info = {
        'file_name': image_name,
        'height': gt_data["imageHeight"],
        'width': gt_data["imageWidth"],
        'id': len(coco_dict['images'])+1
        }
        coco_dict['images'].append(image_info)
    else:
        assert("Invalid ""to_format"". Please check your config file.")
    
    # Load image info
    image_info = gt_data["imageData"]
    img_w, img_h = gt_data["imageWidth"], gt_data["imageHeight"]
    
    # Load object info
    for multi_obj in gt_data["shapes"]:
        #Load object info
        obj_nm = multi_obj["label"]
        # Get object id
        obj_id = classes.index(obj_nm)
        # Get polygon
        polygon = sum(multi_obj["points"], []) 
        # x1, y1, x2, y2, x3, y3, x4, y4 => clockwise
        
        if len(polygon) < 8:
            print("Invalid polygon. len(polygon)<8, Please check your labelme annotation file(%s)"%image_info)
            import pdb; pdb.set_trace()

        # Convert Json to Another Data Format
        if to_t == "yolo":
            new_line = toYolo(args, polygon, img_w, img_h, obj_id, task)
        elif to_t == "dota":
            new_line = toDota(args, polygon, img_w, img_h, obj_nm, task)
        elif to_t == "voc":
            polygon = multi_obj["points"]
            writer = toVoc(args, writer, task, obj_nm, polygon)
        elif to_t == "coco":
            object_info = toCoco(args, polygon, img_w, img_h, obj_nm, obj_id, task, coco_dict)
            coco_dict['annotations'].append(object_info)
        else:
            assert("Invalid ""to_format"". Please check your config file.")
    
        # Save Annotation in output folder
        if to_t in ["yolo", "dota"]:
            out_file.write(new_line)
        elif to_t == "voc":
            # write xml file
            writer.save()
    if to_t == "coco":
        return coco_dict
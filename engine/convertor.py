from tqdm import tqdm
from utils.util import get_class, loadGT
import os
from engine.labelme.labelmeConvert import labelmeConvert
# from engine import YoloConvert
# from engine import VocConvert
# from engine import DotaConvert

class Convertor:
    """_summary_: Convertor class for converting annotation format.
    - Get One pair of gt file, image file. And convert annotation(Save file in output folder)
    - You can add your own convertor by adding your own convertor(format) class in engine folder.
    -
    """
    def __init__(self, args, data_path, from_format, to_format, output_folder, class_file, task="detection"):
        self.from_format = from_format
        self.to_format = to_format
        self.data_path = data_path
        self.classes = get_class(class_file)
        self.output_folder = output_folder
        self.task = task
        
    def run(self, args, image_path, gt_path, output_path, coco_dict=None):
        # convert Annotation
        gt_data = loadGT(self.from_format, gt_path)
        if self.from_format == "labelme":
            coco_dict = labelmeConvert(args, self.to_format, image_path, gt_data, self.classes, output_path, self.task, coco_dict=coco_dict)
        elif self.from_format == "yolo":
            assert("Yolo format is not supported yet.")
            YoloConvert(self.to_format, image_path, gt_data, self.classes, output_path, self.task)
        elif self.from_format == "voc":
            assert("Voc format is not supported yet.")
            VocConvert(self.to_format, image_path, gt_data, self.classes, output_path, self.task)
        elif self.from_format == "dota":
            assert("Dota format is not supported yet.")
            DotaConvert(self.to_format, image_path, gt_data, self.classes, output_path, self.task)
        else:
            print("Invalid ""from_format"". Please check your config file.")
            AssertionError()
            
        return coco_dict
import os
import glob
import xmltodict
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm 
from shutil import copyfile
from utils.util import read_config, getImagesInDir, getGTInDir, getImagesFromFile, getCocoDict, get_class, getGTFromFile
from engine.convertor import Convertor

def main(args):
    data_path = args["data_path"]
    from_format = args["from_format"]
    to_format = args["to_format"]
    output_folder = args["output_folder"]
    class_file = args["classes"]
    classes = get_class(class_file)
    task = args["task"]
    
    # Convertor class
    convertor = Convertor(args, data_path, from_format, to_format, output_folder, class_file, task)
    
    if to_format == "coco":
        # COCO 인 경우에만 Train, Val, Test로 나누어서 처리
        PurposePipeline(args, from_format, to_format, data_path, output_folder, convertor, classes, task)
    else:
        # 여러 문서 종류별로 나열되어 처리하는 경우
        # categorical process
        CategoryPipeline(args, from_format, to_format, data_path, output_folder, convertor)
    
    
def PurposePipeline(args, from_format, to_format, data_path, output_folder, convertor, classes, task):
    purpose_lst = ["train", "val", "test"]
    if args["random_split"]:
        # Val and Test data is same size.
        ratio = [args["train_ratio"], (1-args["train_ratio"])/2, (1-args["train_ratio"])/2]
        print("Randomly split data into train {:.1f}/val {:.1f}/test {:.1f}".format(ratio[0], ratio[1], ratio[2]))
    else:
        # read file
        for purpose in purpose_lst:
            full_path = data_path+purpose
            image_paths = getImagesFromFile(data_path, full_path+".txt")
            gt_paths = getGTFromFile(data_path, from_format, image_paths)
            output_path = output_folder+"/"+purpose
            
            if not os.path.exists(output_path+'/images'):
                os.makedirs(output_path+'/images')
            if not os.path.exists(output_path+'/annotations'):
                os.makedirs(output_path+'/annotations')
                
            file_nm_lst = open(output_path+".txt", "w")
            # Only for Coco Format
            # You must modify "getCocoDict" if you want to use your own dataset.
            output_json_dict, category_dict = getCocoDict(classes, task)
            output_json_dict["categories"] = category_dict
            
            for image_path, gt_path in tqdm(zip(image_paths, gt_paths),\
                                            desc="Changing - [%s]"%purpose, \
                                            total=len(image_paths)):
                
                # Check image-gt pair is same.
                if image_path != gt_path:
                    AssertionError("Image and GT file is not same. Please check your data.")
                file_nm_lst.write(image_path + '\n')
                # Convert Annotation as Yolo TXT format
                
                output_json_dict = convertor.run(args, image_path, gt_path, output_path, coco_dict=output_json_dict)
                # Copy Image files
                copyfile(image_path, output_path+'/images/'+image_path.split("/")[-1])
            file_nm_lst.close()
            output_json_path = output_path+"/annotations/"+purpose+".json"
            with open(output_json_path, 'w', encoding='utf-8-sig') as f:
                output_json = json.dumps(output_json_dict, ensure_ascii=False, indent='\t')
                f.write(output_json)

def CategoryPipeline(args, from_format, to_format, data_path, output_folder, convertor):
    subdir_lst = [f for f in os.listdir(data_path) if not f.startswith('.') and not f.endswith('.txt')]
    
    for subdir in subdir_lst:
        full_path = data_path+subdir
        image_paths = getImagesInDir(full_path)
        gt_paths = getGTInDir(full_path, from_format, image_paths)
        output_path = output_folder+"/"+subdir
        
        if not os.path.exists(output_path+'/images'):
            os.makedirs(output_path+'/images')
        if not os.path.exists(output_path+'/labels'):
            os.makedirs(output_path+'/labels')
            
        file_nm_lst = open(output_path+".txt", "w")
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths),\
                                        desc="Changing - [%s]"%subdir, \
                                        total=len(image_paths)):
            
            # Check image-gt pair is same.
            if image_path != gt_path:
                AssertionError("Image and GT file is not same. Please check your data.")
            file_nm_lst.write(image_path + '\n')
            # Convert Annotation as Yolo TXT format
            output = convertor.run(args, image_path, gt_path, output_path)
            # Copy Image files
            copyfile(image_path, output_path+'/images/'+image_path.split("/")[-1])
        file_nm_lst.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='path to config file')
    args = parser.parse_args()
    args = read_config(args.config)
    
    if args["from_format"] not in ["labelme", "yolo", "coco", "voc", "dota"] or args["to_format"] not in ["labelme", "yolo", "coco", "voc", "dota"]:
        raise ValueError("Invalid from_format. Please check your config file.")
    
    main(args)
# DataForm_Convertor

- Currently only support labelme to other format(YOLO, COCO, PascalVOC, Dota)


### Config File
```
# Available Data type : ["labelme", "voc", "coco", "yolo", "dota"]
from_format: "labelme" #input format you want to change as format "TO".
to_format: "coco" #output data format changed from FROM
# YOLO, VOC, Dota Complete 

data_path: "./data/labelme/" #data folder OR data list file
classes: "./data/labels.txt" #txt file with class label
output_folder: "./outputs" #path to output folder
task: "detect" # ["detect", "segm", "keypoint"]

# CoCo format
random_split: False # (Bool) 
train_ratio: 0.7 #Ratio of Train-Val-Test(Val : test == 1:1)

# Dota
normalize: True # normalize the coordinate by image size(w, h)
```

### Data set structure
```
📦data
 ┣ 📂labelme
 ┃  ┣ 📂{DIR_1}
 ┃  ┃ ┣ 📜BT_t0.json
 ┃  ┃ ┣ 📜BT_t0.png
 ┃  ┃ ┣ ...
 ┃  ┃ ┣ 📜BT_tN.json
 ┃  ┃ ┗ 📜BT_tN.png
 ┃  ┣ 📂{DIR_2}
 ┃  ┣ 📂{DIR_3}
 ┃  ┣ 📂{DIR_4}
 ┃  ┣ ...
 ┃  ┣ 📂{DIR_N-1}
 ┃  ┣ 📂{DIR_N}
 ┃  ┣ 📜test.txt
 ┃  ┣ 📜train.txt
 ┃  ┗ 📜val.txt
 ┗ 📜labels.txt
```

- train/val/test.txt is related like below
```
# list of image sets.
{DIR_1}/BT_t0.png
{DIR_1}/BT_t1.png
{DIR_1}/BT_t2.png
...
{DIR_4}/BT_tN.png
```

### Running
```
# install
$ pip install -r requirement.txt
$ python main.py
```
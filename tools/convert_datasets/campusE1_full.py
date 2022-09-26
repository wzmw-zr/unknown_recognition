import argparse
import os.path as osp
import mmcv
import json
from PIL import Image
from PIL import ImageDraw
from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .

    'trainId'     , # Max value is 255!

    'color'       , # The color of this label

    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

labels = [
    #       name                    trainId    color
    Label(  'road'                 ,        0 , (128, 64,128) ),
    Label(  'sidewalk'             ,        1 , (244, 35,232) ),
    Label(  'building'             ,        2 , ( 70, 70, 70) ),
    Label(  'wall'                 ,        3 , (102,102,156) ),
    Label(  'fence'                ,        4 , (190,153,153) ),
    Label(  'pole'                 ,        5 , (153,153,153) ),
    Label(  'light'        ,        6 , (250,170, 30) ),
    Label(  'sign'         ,        7 , (220,220,  0) ),
    Label(  'vegetation'           ,        8 , (107,142, 35) ),
    Label(  'terrain'              ,        9 , (152,251,152) ),
    Label(  'sky'                  ,       10 , ( 70,130,180) ),
    Label(  'person'               ,       11 , (220, 20, 60) ),
    Label(  'rider'                ,       12 , (255,  0,  0) ),
    Label(  'car'                  ,       13 , (  0,  0,142) ),
    Label(  'truck'                ,       14 , (  0,  0, 70) ),
    Label(  'bus'                  ,       15 , (  0, 60,100) ),
    Label(  'train'                ,       16 , (  0, 80,100) ),
    Label(  'motorcycle'           ,       17 , (  0,  0,230) ),
    Label(  'bicycle'              ,       18 , (119, 11, 32) ),
    Label(  'staircase'            ,     19,  (255,224,  0)),
    Label(    'curb'               ,      20,  ( 82,  0,255)),
    Label(    'ramp'               ,     21,  (255,245,  0)),
    Label(  'runway'               ,     22,  (153,255,  0)),
    Label(  'flowerbed'            ,     23,  (160,150, 20)),
    Label(    'door'               ,     24,  (  8,255, 51)),
    Label(   'camera'              ,     25,  (  0,204,255)),
    Label(   'manhole'             ,     26,  ( 80, 50, 50)),
    Label(   'hydrant'             ,     27,  (  0,204,255)),
    Label(    'belt'               ,     28,  (255, 51,  7)),
    Label(  'dustbin'              ,     29,  (  0,163,255)),
    Label(     'lawn'              ,    30,  (  4,250,  7)),

    Label(    'pipe'               ,    255,  ( 0, 0, 0)),
    Label(    'chair'              ,    255,  (0, 0,  0)),
    Label(    'bench'              ,    255,  (0, 0,  0)),
    Label(    'others'             ,    255,  (  0,  0,  0)),
    Label(  '__ignore__'           ,    255,  (  0,  0,  0)), 
    Label(  'unlabeled'            ,    255,  (  0,  0,  0)), 
]

label_set = set()

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

chinese_to_english = {
    '建筑': "building", 
    '人行路': "sidewalk", 
    '其他': "others", 
    '标识牌': "sign", 
    '隔离带': "belt", 
    '其它': "others", 
    '摄像头': "camera", 
    '柱子': "pole", 
    '树灌木丛': "vegetation", 
    '跑道': "runway", 
    '门': "door", 
    '路缘石': "curb", 
    '井盖': "manhole", 
    '马路': "road", 
    '栅栏': "fence", 
    '椅子': "chair", 
    '树、灌木丛等': "vegetation", 
    '水管': "pipe", 
    '其他类': "others", 
    '消防栓': "hydrant", 
    '花坛': "flowerbed", 
    '草坪': "lawn", 
    '\ufeff马路': "road", 
    '天空': "sky", 
    '坡道': "ramp", 
    '行人': "person", 
    '汽车': "car", 
    '楼梯': "staircase", 
    '墙': "wall", 
    '广告牌': "sign", 
    '巴士': "car", 
    '长凳': "bench", 
    '垃圾箱': "dustbin"
}

# name to label object
name2label = { label.name : label for label in labels }

class Polygon(object):
    def __init__(self, jsonText) -> None:
        # points consists of the polygon
        self.label = jsonText["label"]
        if self.label in chinese_to_english.keys():
            self.label = chinese_to_english[self.label]
        self.polygon = [Point(p[0], p[1]) for p in jsonText["points"]]


class Annotation(object):
    """The annotation of a whole image"""
    def __init__(self, jsonText):
        # the height of that image and thus of the label image
        self.imageHeight = jsonText["imageHeight"]
        # the width of that image and thus of the label image
        self.imageWidth = jsonText["imageWidth"]
        # the list of objects
        self.objects = [Polygon(obj) for obj in jsonText["shapes"]]


def createLabelImage(annotation, encoding, outline=None):
    # the size of the image

    size = ( annotation.imageWidth , annotation.imageHeight )

    # the background
    if encoding == "trainIds":
        background = name2label['unlabeled'].trainId
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        if not label in name2label:
            print(f"Label {label} not known.")

        if encoding == "trainIds":
            val = name2label[label].trainId
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:
                drawer.polygon( polygon, fill=val, outline=outline )
            else:
                drawer.polygon( polygon, fill=val )
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    return labelImg


def json2labelImg(json_file, out_image, encoding="trainIds"):
    """ A method that does all the work
    Args:    
        - json_file: the filename of the json file
        - out_image: the filename of the label image that is generated
        - encoding:
            - "trainIds" : classes are encoded using the training IDs
            - "color"    : classes are encoded using the corresponding colors
    """
    with open(json_file, "r") as f:
        json_text = json.load(f)
    annotation = Annotation(json_text)
    labelImg = createLabelImage(annotation, encoding)
    labelImg.save(out_image)


def convert_json_to_label(json_file: str):
    label_file = json_file.replace('.json', '_labelTrainIds.png')
    label_file = label_file.replace(gtFine_dir, out_dir + "/")
    mmcv.mkdir_or_exist(osp.dirname(label_file))
    print(f"process {json_file}, generate {label_file}")
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CampusE1 annotations to TrainIds")
    parser.add_argument('gtFine_dir', help="gtFine directory of campus dataset")
    parser.add_argument('-o', '--out-dir', help="output path")
    parser.add_argument(
        "--nproc", default=1, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    global gtFine_dir
    global out_dir
    gtFine_dir = args.gtFine_dir
    out_dir = args.out_dir if args.out_dir else gtFine_dir
    mmcv.mkdir_or_exist(out_dir)


    json_files = []

    for json_file in mmcv.scandir(gtFine_dir, '.json', recursive=True):
        json_file = osp.join(gtFine_dir, json_file)
        json_files.append(json_file)
    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_json_to_label, json_files,
                                     args.nproc)
    else:
        mmcv.track_progress(convert_json_to_label, json_files)


if __name__ == "__main__":
    main()
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
    Label(   'road'                ,      0,  (128, 64,128)),
    Label(  'sidewalk'             ,      1,  (244, 35,232)),
    Label(    'curb'               ,    255,  ( 82,  0,255)),

    Label(  'building'             ,    255,  ( 70, 70, 70)),
    Label(     'wall'              ,    255,  (102,102,156)),
    Label(    'fence'              ,    255,  (190,153,153)),
    Label(     'pole'              ,    255,  (153,153,153)),
    Label(  'runway'               ,    255,  (153,255,  0)),
    Label(   'bench'               ,    255,  (194,255,  0)),
    Label(  'vegetation'           ,    255,  (107,142, 35)),
    Label(    'pipe'               ,    255,  ( 11,102,255)),
    Label(    'sky'                ,    255,  ( 70,130,180)),
    Label( 'pedestrian'            ,    255,  (220, 20, 60)),
    Label(    'door'               ,    255,  (  8,255, 51)),
    Label(    'car'                ,    255,  (  0,  0,142)),
    Label(  'dustbin'              ,    255,  (  0,163,255)),
    Label(   'camera'              ,    255,  (  0,204,255)),
    Label(  'flowerbed'            ,    255,  (160,150, 20)),
    Label(    'ramp'               ,    255,  (255,245,  0)),
    Label(    'belt'               ,    255,  (255, 51,  7)),
    Label(    'chair'              ,    255,  (204, 70,  3)),
    Label(   'manhole'             ,    255,  ( 80, 50, 50)),
    Label(     'lawn'              ,    255,  (  4,250,  7)),
    Label(     'sign'              ,    255,  (220,220,  0)),
    Label(  'staircase'            ,    255,  (255,224,  0)),
    Label(   'hydrant'             ,    255,  (  0,204,255)),
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
    '行人': "pedestrian", 
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
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
    Label(    'curb'               ,      1,  ( 82,  0,255)),
    Label(  'building'             ,      2,  ( 70, 70, 70)),
    Label(     'wall'              ,      3,  (102,102,156)),
    Label(    'fence'              ,      4,  (190,153,153)),
    Label(     'pole'              ,      5,  (153,153,153)),
    Label(  'runway'               ,      6,  (153,255,  0)),
    Label(   'bench'               ,      7,  (194,255,  0)),
    Label(  'vegetation'           ,      8,  (107,142, 35)),
    Label(    'pipe'               ,      9,  ( 11,102,255)),
    Label(    'sky'                ,     10,  ( 70,130,180)),
    Label( 'pedestrian'            ,     11,  (220, 20, 60)),
    Label(    'door'               ,     12,  (  8,255, 51)),
    Label(    'car'                ,     13,  (  0,  0,142)),

    Label(  'dustbin'              ,     14,  (  0,163,255)),
    Label(   'camera'              ,     15,  (  0,204,255)),
    Label(  'flowerbed'            ,     16,  (160,150, 20)),
    Label(    'ramp'               ,     17,  (255,245,  0)),
    Label(    'belt'               ,     18,  (255, 51,  7)),
    Label(    'chair'              ,     19,  (204, 70,  3)),
    Label(   'manhole'             ,     20,  ( 80, 50, 50)),
    Label(     'lawn'              ,     21,  (  4,250,  7)),
    Label(     'sign'              ,     22,  (220,220,  0)),
    Label(  'staircase'            ,     24,  (255,224,  0)),
    Label(   'hydrant'             ,     25,  (  0,204,255)),
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
    '??????': "building", 
    '?????????': "sidewalk", 
    '??????': "others", 
    '?????????': "sign", 
    '?????????': "belt", 
    '??????': "others", 
    '?????????': "camera", 
    '??????': "pole", 
    '????????????': "vegetation", 
    '??????': "runway", 
    '???': "door", 
    '?????????': "curb", 
    '??????': "manhole", 
    '??????': "road", 
    '??????': "fence", 
    '??????': "chair", 
    '??????????????????': "vegetation", 
    '??????': "pipe", 
    '?????????': "others", 
    '?????????': "hydrant", 
    '??????': "flowerbed", 
    '??????': "lawn", 
    '\ufeff??????': "road", 
    '??????': "sky", 
    '??????': "ramp", 
    '??????': "pedestrian", 
    '??????': "car", 
    '??????': "staircase", 
    '???': "wall", 
    '?????????': "sign", 
    '??????': "car", 
    '??????': "bench", 
    '?????????': "dustbin"
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
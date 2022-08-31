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
    Label(  'Road'                 ,      0 ,  (128, 64,128) ),
    Label(  'Sidewalk'             ,      1 ,  (244, 35,232) ),
    Label(  'Buildings'            ,      2 ,  ( 70, 70, 70) ),
    Label(  'Walls'                ,      3 ,  (102,102,156) ),
    Label(  'Fence'                ,      4 ,  (190,153,153) ),
    # Label(  'pole'                 ,      5 ,  (153,153,153) ),
    # Label(  'traffic light'        ,      6 ,  (250,170, 30) ),
    # Label(  'traffic sign'         ,      7 ,  (220,220,  0) ),
    # Label(  'vegetation'           ,      8 ,  (107,142, 35) ),
    # Label(  'terrain'              ,      9 ,  (152,251,152) ),
    Label(  'Sky'                  ,     10 ,  ( 70,130,180) ),
    # Label(  'person'               ,     11 ,  (220, 20, 60) ),
    # Label(  'rider'                ,     12 ,  (255,  0,  0) ),
    # Label(  'car'                  ,     13 ,  (  0,  0,142) ),
    # Label(  'truck'                ,     14 ,  (  0,  0, 70) ),
    Label(  'Car'                  ,     13,  (  0, 60,100) ),
    Label(  'Bus'                  ,     15,  (  0, 60,100) ),
    # Label(  'Bus'                  ,     15 ,  (  0, 60,100) ),
    # Label(  'train'                ,     16 ,  (  0, 80,100) ),
    # Label(  'motorcycle'           ,     17 ,  (  0,  0,230) ),
    # Label(  'bicycle'              ,     18 ,  (119, 11, 32) ),
    Label(  '__ignore__'           ,    255 ,  (  0,  0,  0) ), 
    Label(  'unlabeled'            ,    255 ,  (  0,  0,  0) ), 
    Label(  'Trees'                ,      8 ,  (107,142, 35) ),
    Label(  'FlowerBeds'           ,      8 ,  (107,142, 35) ),
    Label(  'Lawn'                 ,      8 ,  (111, 74,  0) ), 
    Label(  'Staircase'            ,     19 ,  (111, 74,  0) ), 
    Label(  'ManholeCover'         ,     19 ,  (111, 74,  0) ), 
    Label(  'Chairs'               ,     19 ,  (111, 74,  0) ), 
    Label(  'Curbs'                ,     19 ,  (111, 74,  0) ), 
    # Label(  'Lawn'                 ,     19 ,  (111, 74,  0) ), 
    Label(  'Posts'                ,     19 ,  (111, 74,  0) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = { label.name : label for label in labels }

class Polygon(object):
    def __init__(self, jsonText) -> None:
        # points consists of the polygon
        self.label = jsonText["label"]
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


def createLabelImage(json_file, annotation, encoding, outline=None):
    # the size of the image
    json_file_name = json_file.split("/")[-1]
    if json_file_name in change_label_images.keys():
        changes = change_label_images[json_file_name]
    else:
        changes = None

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

        if changes is not None:
            if label in changes:
                label = changes[label]

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
    labelImg = createLabelImage(json_file, annotation, encoding)
    labelImg.save(out_image)


def convert_json_to_label(json_file: str):
    # json_file_name = json_file.split("/")[-1]
    # if json_file_name not in valid_images and json_file_name not in change_label_images.keys():
    #     print(f"{json_file_name} is excluded.")
    #     return 
    label_file = json_file.replace('.json', '_labelTrainIds.png')
    print(f"process {json_file}")
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
    gtFine_dir = args.gtFine_dir
    out_dir = args.out_dir if args.out_dir else gtFine_dir
    mmcv.mkdir_or_exist(out_dir)

    with open("valid_images.json", "r") as f:
        global valid_images
        valid_images = json.load(f)
    with open("change_label_images.json", "r") as f:
        global change_label_images
        change_label_images = json.load(f)
    print(valid_images)
    print(change_label_images.keys())

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
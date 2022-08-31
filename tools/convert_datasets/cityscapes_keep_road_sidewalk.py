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

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                      trainId   color
    Label(  'unlabeled'            ,      255 , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,      255 , (  0,  0,  0) ),
    Label(  'rectification border' ,      255 , (  0,  0,  0) ),
    Label(  'out of roi'           ,      255 , (  0,  0,  0) ),
    Label(  'static'               ,      255 , (  0,  0,  0) ),
    Label(  'dynamic'              ,      255 , (111, 74,  0) ),
    Label(  'ground'               ,      255 , ( 81,  0, 81) ),
    Label(  'road'                 ,        0 , (128, 64,128) ),
    Label(  'sidewalk'             ,        1 , (244, 35,232) ),
    Label(  'parking'              ,      255 , (250,170,160) ),
    Label(  'rail track'           ,      255 , (230,150,140) ),
    Label(  'building'             ,      255 , ( 70, 70, 70) ),
    Label(  'wall'                 ,      255 , (102,102,156) ),
    Label(  'fence'                ,      255 , (190,153,153) ),
    Label(  'guard rail'           ,      255 , (180,165,180) ),
    Label(  'bridge'               ,      255 , (150,100,100) ),
    Label(  'tunnel'               ,      255 , (150,120, 90) ),
    Label(  'pole'                 ,      255 , (153,153,153) ),
    Label(  'polegroup'            ,      255 , (153,153,153) ),
    Label(  'traffic light'        ,      255 , (250,170, 30) ),
    Label(  'traffic sign'         ,      255 , (220,220,  0) ),
    Label(  'vegetation'           ,      255 , (107,142, 35) ),
    Label(  'terrain'              ,      255 , (152,251,152) ),
    Label(  'sky'                  ,      255 , ( 70,130,180) ),
    Label(  'person'               ,      255 , (220, 20, 60) ),
    Label(  'rider'                ,      255 , (255,  0,  0) ),
    Label(  'car'                  ,      255 , (  0,  0,142) ),
    Label(  'truck'                ,      255 , (  0,  0, 70) ),
    Label(  'bus'                  ,      255 , (  0, 60,100) ),
    Label(  'caravan'              ,      255 , (  0,  0, 90) ),
    Label(  'trailer'              ,      255 , (  0,  0,110) ),
    Label(  'train'                ,      255 , (  0, 80,100) ),
    Label(  'motorcycle'           ,      255 , (  0,  0,230) ),
    Label(  'bicycle'              ,      255 , (119, 11, 32) ),
    Label(  'license plate'        ,      255 , (  0,  0,  0) ),
]

# name to label object
name2label = { label.name : label for label in labels }

class Polygon(object):
    def __init__(self, jsonText) -> None:
        # points consists of the polygon
        self.label = jsonText["label"]
        self.polygon = [Point(p[0], p[1]) for p in jsonText["polygon"]]


class Annotation(object):
    """The annotation of a whole image"""
    def __init__(self, jsonText):
        # the height of that image and thus of the label image
        self.imageHeight = jsonText["imgHeight"]
        # the width of that image and thus of the label image
        self.imageWidth = jsonText["imgWidth"]
        # the list of objects
        self.objects = [Polygon(obj) for obj in jsonText["objects"]]


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

        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]

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
        description="Convert cityscapes annotations to TrainIds")
    parser.add_argument('gtFine_dir', help="gtFine directory of cityscapes dataset")
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
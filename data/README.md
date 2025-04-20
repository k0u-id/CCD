# Getting the Data

## PASCAL

1. Navigate to the PASCAL data directory:
```
cd /path/to/CCD/data/pascal
```
2. Download the data:
```
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_12.tar
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar --output pascal_07.tar
```
3. Extract the data:
```
tar -xf pascal_12.tar
tar -xf pascal_07.tar
```
4. Clean up:
```
rm pascal_12.tar
rm pascal_07.tar
```

## COCO

1. Navigate to the COCO data directory:
```
cd /path/to/CCD/data/coco
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Clean up:
```
rm coco_train_raw.zip
rm coco_val_raw.zip
```

## NUSWIDE

*These instructions differ slightly from those for the other datasets because we re-crawled NUSWIDE.*

1. Follow the instructions [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to request a download link for the NUSWIDE images. Once approved, you will receive a link to download `Flickr.zip` which contains the images for the NUSWIDE dataset. Download this file and move it to the NUSWIDE data directory, so that the full path is:
```
/path/to/CCD/data/nuswide/Flickr.zip
```
2. Navigate to the NUSWIDE data directory:
```
cd /path/to/CCD/data/nuswide
```
3. Extract the images:
```
unzip -q Flickr.zip
```
4. Clean up:
```
rm Flickr.zip
```

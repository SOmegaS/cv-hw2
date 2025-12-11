#!/bin/bash

# Script to download COCO dataset subset
# Downloads train2017, val2017, and annotations

COCO_DIR="./data/coco"
mkdir -p $COCO_DIR

echo "Downloading COCO dataset..."
echo "This may take a while (about 20GB)"

cd $COCO_DIR

# Download train2017
if [ ! -d "train2017" ]; then
    echo "Downloading train2017 images..."
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
else
    echo "train2017 already exists, skipping..."
fi

# Download val2017
if [ ! -d "val2017" ]; then
    echo "Downloading val2017 images..."
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
else
    echo "val2017 already exists, skipping..."
fi

# Download annotations
if [ ! -d "annotations" ]; then
    echo "Downloading annotations..."
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo "annotations already exists, skipping..."
fi

echo "COCO dataset download complete!"
echo "Dataset location: $COCO_DIR"


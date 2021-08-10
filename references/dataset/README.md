# Dataset Constructs

### Dataset construct for SSD and FRMN (fruitDefectDataset.py)

This the class construct code that we used to create the dataset

Note: After loading in all the data, we create a dictionary that links every
image id to the the label and bounding box information. We passed this data
to this class.

To utilize this class for further datasets. Just construct a
* id_labels: {"image.jpeg", [1, 0], ...} where the nums are class ids
* id_bounding_boxes {"image.jpeg": [[10, 20, 50, 100], [30, 40, 70, 87]]} where the lists are bbox values.

### Dataset construct for EfficientDET (effDET_fruitDefectDataset.py)

Input:
* labels_dict_fn: {"image.jpeg", [1, 0], ...} where the nums are class ids
* bounding_box_dict_fn: {"image.jpeg": [[10, 20, 50, 100], [30, 40, 70, 87]]} where the lists are bbox values.
* imgs_key_fn: image.jpeg
Returns: It converts the labels dict & bbox dict data to a dictionary in coco format that can be dumped into a json file. This json file is needed in training the effecientdet scripts as shown here: https://github.com/rwightman/efficientdet-pytorch

### Dataset construct for Noise Dataset (noiseDataset.py)

OPTIONAL: This is the class construct for a test file our paper used to guage how the model did on not predicting anything on random images.

It is not needed to run and construct entire pipeline





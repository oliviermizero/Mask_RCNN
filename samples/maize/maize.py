"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == "__main__":
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import os
import os.path as osp
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/maize/")


############################################################
#  Configurations
############################################################


class MaizeConfig(Config):
    """Configuration for training on the maize segmentation dataset."""

    # Give the configuration a recognizable name
    NAME = "maize"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Kernel + Ear

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 1
    VALIDATION_STEPS = 1

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 170

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 500

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 500


class MaizeInferenceConfig(MaizeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################
VAL_IMAGE_IDS = []
TEST_IMAGE_IDS = []


class MaizeDataset(utils.Dataset):
    def load_maize(self, dataset_dir, subset):

        # Add the class names using the base method from utils.Dataset
        """Implement way to add classes from dataset"""
        self.add_class("Maize", 1, "kernel")
        self.add_class("Maize", 2, "cob")

        # Get filenames and annotation
        filenames = next(os.walk(dataset_dir))[2]
        annotation_filename = [
            filename for filename in filenames if filename.split(".")[1] == "json"
        ][0]

        # Which subset?
        # "val":
        # "train":
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        elif subset == "test":
            image_ids = TEST_IMAGE_IDS
        else:
            image_ids = [
                filename.split(".")[0]
                for filename in filenames
                if (
                    (len(filename.split("_")) <= 4)
                    and (filename.split(".")[1] == "png")
                )
            ]
            image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        annotation_json = osp.join(dataset_dir, annotation_filename)
        with open(annotation_json) as f:
            annotations = json.load(f)

        # Add images
        for image_id in image_ids:
            image = image_id + ".png"
            # Get class ids
            for annotation in annotations:
                if annotation["image_name"] == image_id:
                    class_ids = annotation["class_ids"]
                    # continue

            self.add_image(
                "Maize",
                image_id=image_id,
                path=os.path.join(dataset_dir, image),
                class_ids=class_ids,
            )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        image_name = info["id"]
        bitmap_path = osp.join(
            (osp.dirname(info["path"])), f"{image_name}_label_ground-truth.png"
        )
        instances = info["class_ids"]
        # Read mask files from .png image
        masks = []
        class_ids = []
        bitmap = skimage.io.imread(bitmap_path)
        for instance in instances:
            m = np.array(bitmap, np.uint32) == instance["id"]
            masks.append(m)
            class_ids.append(instance["class_id"])

        # Return mask, and array of class IDs of each instance.
        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Maize":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


def train(model, config, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MaizeDataset()
    dataset_train.load_maize(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MaizeDataset()
    dataset_val.load_maize(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf(
        (0, 2),
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf(
                [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]
            ),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ],
    )

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=20,
        augmentation=augmentation,
        layers="heads",
    )

    print("Train all layers")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=40,
        augmentation=augmentation,
        layers="all",
    )


############################################################
#  Detection
############################################################


def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MaizeDataset()
    dataset.load_maize(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]

        # Encode image to RLE. Returns a string of multiple lines
        # source_id = dataset.image_info[image_id]["id"]
        # rle = mask_to_rle(source_id, r["masks"], r["scores"])
        # submission.append(rle)

        # Save image with maskss
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            dataset.class_names,
            r["scores"],
            show_bbox=False,
            show_mask=False,
            title="Predictions",
        )
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for kernel counting and segmentation"
    )
    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/dataset/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--subset",
        required=False,
        metavar="Dataset sub-directory",
        help="Subset of dataset to run prediction on",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MaizeConfig()
    else:
        config = MaizeInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. " "Use 'train' or 'detect'".format(args.command))


if __name__ == "__main__":
    main()

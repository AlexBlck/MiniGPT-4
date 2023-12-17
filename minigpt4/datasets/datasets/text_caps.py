import itertools
import json
import os
import pickle
import random
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import webdataset as wds
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class VixenDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Describe the defferences between the two images.",
            "What are the differences between the two images?",
            "What is different between the two images?",
            "The differences between the two images are:",
            "List all the edits made to the image.",
            "List all the changes made to the image.",
            "How is the second image different from the first?",
            "How was the first image changed to make the second image?",
            "What changes were made to the first image to make the second image?",
            "How was the first image manipulated to make the second image?",
            "Summarize the changes made to the image.",
        ]

        self.impaths = torch.load(join(vis_root, "train_files.pt"))

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, index):
        image_file1 = self.impaths[index]
        image_file2 = image_file1.replace("_0.jpg", "_1.jpg")
        image_path1 = join(self.vis_root, image_file1)
        image_path2 = join(self.vis_root, image_file2)
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")
        image1 = self.vis_processor(image1)
        image2 = self.vis_processor(image2)

        with open(join(image_file1.split("/")[0], "prompt_davinci.json"), "r") as f:
            data = json.load(f)

        caption = data["caption_curie"]

        caption = self.text_processor(caption)
        instruction = f"<Img><ImageHere></Img> <Img><ImageHere></Img> [idc] {random.choice(self.instruction_pool)} "
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }


class TextCapDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Briefly describe this image.",
            "Provide a concise depiction of this image.",
            "Present a short description of this image.",
            "Summarize this image in a few words.",
            "A short image caption:",
            "A short image description:",
            "A photo of ",
            "An image that shows ",
            "Write a short description for the image. ",
            "Write a description for the photo.",
            "Provide a description of what is presented in the photo.",
            "Briefly describe the content of the image.",
            "Can you briefly explain what you see in the image?",
            "Could you use a few words to describe what you perceive in the photo?",
            "Please provide a short depiction of the picture.",
            "Using language, provide a short account of the image.",
            "Use a few words to illustrate what is happening in the picture.",
        ]

        with open(ann_path, "r") as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann["data"])

    def __getitem__(self, index):
        info = self.ann["data"][index]

        image_file = "{}.jpg".format(info["image_id"])

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = info["caption_str"]
        caption = self.text_processor(caption)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(
            random.choice(self.instruction_pool)
        )
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
        }

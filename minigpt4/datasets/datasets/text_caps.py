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
from datasets import load_dataset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class MetsDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.num_imgs = 2
        self.use_text = True

        self.instruction_pool = [
            "Describe the defferences between the images.",
            "What are the differences between the images?",
            "What is different between the images?",
            "The differences between the images are:",
            "List all the edits made to the image.",
            "List all the changes made to the image.",
            "How is the last image different from the first?",
            "How was the first image changed to make the last image?",
            "What changes were made to the first image to make the last image?",
            "How was the first image manipulated to make the last image?",
            "Summarize the changes made to the image.",
        ]

        self.captions = torch.load(join(vis_root, "mets_train.pt"))
        self.captions = [(k, v) for k, v in self.captions.items()]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        try:
            image_file2, caption = self.captions[index]
            if self.num_imgs > 0:
                max_idx = int(image_file2[-6:-4])
                img_ids = np.linspace(0, max_idx, self.num_imgs, dtype=int)

                imgs = []
                for img_id in img_ids:
                    image_file = image_file2[:-6] + f"{img_id:02d}.png"
                    image_path = join(self.vis_root, image_file)
                    image = Image.open(image_path).convert("RGB")
                    image = self.vis_processor(image)
                    imgs.append(image)

                image = torch.stack(imgs, dim=0)

                caption = self.text_processor(caption)
                imgs_instruction = " ".join(
                    ["<Img><ImageHere></Img>" for _ in range(self.num_imgs)]
                )
            else:
                imgs_instruction = ""
                image = torch.zeros(0)

            if self.use_text:
                text_filepath = image_file2[:-9] + ".txt"
                lines = open(join(self.vis_root, text_filepath), "r").readlines()
                text_instruction = "edits list: " + " ".join(lines)
            else:
                text_instruction = ""

            instruction = f"{imgs_instruction} {text_instruction} [idc] {random.choice(self.instruction_pool)} "
            return {
                "image": image,
                "instruction_input": instruction,
                "answer": caption,
                "length": self.num_imgs,
            }
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.captions) - 1))


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
        image = torch.stack([image1, image2], dim=0)

        with open(
            join(self.vis_root, image_file1.split("/")[0], "prompt_davinci.json"), "r"
        ) as f:
            data = json.load(f)

        caption = data["caption_curie"]

        caption = self.text_processor(caption)
        instruction = f"<Img><ImageHere></Img> <Img><ImageHere></Img> [idc] {random.choice(self.instruction_pool)} "
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "length": 2,
        }


class CLEVRDataset(Dataset):
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

        change_captions = join(vis_root, "change_captions.json")
        with open(change_captions) as f:
            change_captions = json.load(f)

        nochange_captions = join(vis_root, "no_change_captions.json")
        with open(nochange_captions) as f:
            nochange_captions = json.load(f)

        self.change_captions = {
            int(k.replace(".png", "").split("_")[-1]): change_captions[k]
            for k in change_captions
        }
        self.nochange_captions = {
            int(k.replace(".png", "").split("_")[-1]): nochange_captions[k]
            for k in nochange_captions
        }

        splits = join(vis_root, "splits.json")
        with open(splits) as f:
            splits = json.load(f)

        self.ids = splits["train"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]
        image_file1 = join(self.vis_root, "images", f"CLEVR_default_{index:06d}.png")
        if random.random() < 1.5:
            image_file2 = join(
                self.vis_root, "sc_images", f"CLEVR_semantic_{index:06d}.png"
            )
            caption = random.choice(self.change_captions[index])
        else:
            image_file2 = join(
                self.vis_root, "nsc_images", f"CLEVR_nonsemantic_{index:06d}.png"
            )
            caption = random.choice(self.nochange_captions[index])

        caption = self.text_processor(caption)
        image1 = Image.open(image_file1).convert("RGB")
        image2 = Image.open(image_file2).convert("RGB")
        image1 = self.vis_processor(image1)
        image2 = self.vis_processor(image2)
        image = torch.stack([image1, image2], dim=0)

        instruction = f"<Img><ImageHere></Img> <Img><ImageHere></Img> [idc] {random.choice(self.instruction_pool)} "
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "length": 2,
            "path": image_file1,
        }


class MagicBrushChain(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.dataset = load_dataset("osunlp/MagicBrush", cache_dir=vis_root)["train"]
        self.info = self.dataset.remove_columns(
            ["source_img", "target_img", "mask_img"]
        )
        self.ids = np.unique(self.info["img_id"])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Give instructions to edit the image.",
            "How would you edit the image?",
            "What edits should I do no make the second image from the first?",
            "Image editing instructions:",
            "Image editing requests:",
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]
        idxs = np.where([x == index for x in self.info["img_id"]])[0]
        if len(idxs) < 3:
            return self.__getitem__(random.randint(0, len(self) - 1))

        imgs = []
        for idx in idxs:
            image = self.dataset[int(idx)]["source_img"].convert("RGB")
            image = self.vis_processor(image)
            imgs.append(image)

        image = self.dataset[int(idxs[-1])]["target_img"].convert("RGB")
        imgs.append(self.vis_processor(image))
        image = torch.stack(imgs, dim=0)

        caption = [
            f"{i}: {self.info[int(idxs[i])]['instruction']}." for i in range(len(idxs))
        ]
        caption = "\n".join(caption)
        caption = self.text_processor(caption)

        imgs_instruction = " ".join(
            ["<Img><ImageHere></Img>" for _ in range(len(idxs) + 1)]
        )
        instruction = (
            f"{imgs_instruction} [idc] {random.choice(self.instruction_pool)} "
        )
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "length": len(idxs) + 1,
        }


class MagicBrushFirstLast(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.dataset = load_dataset("osunlp/MagicBrush", cache_dir=vis_root)["train"]
        self.info = self.dataset.remove_columns(
            ["source_img", "target_img", "mask_img"]
        )
        self.ids = np.unique(self.info["img_id"])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Give instructions to edit the image.",
            "How would you edit the image?",
            "What edits should I do no make the second image from the first?",
            "Image editing instructions:",
            "Image editing requests:",
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]
        idxs = np.where([x == index for x in self.info["img_id"]])[0]

        image1 = self.dataset[int(idxs[0])]["source_img"].convert("RGB")
        image2 = self.dataset[int(idxs[-1])]["target_img"].convert("RGB")
        image1 = self.vis_processor(image1)
        image2 = self.vis_processor(image2)
        image = torch.stack([image1, image2], dim=0)

        caption = [
            f"{i}: {self.info[int(idxs[i])]['instruction']}." for i in range(len(idxs))
        ]
        caption = "\n".join(caption)
        caption = self.text_processor(caption)

        instruction = f"<Img><ImageHere></Img> <Img><ImageHere></Img> [idc] {random.choice(self.instruction_pool)} "
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "length": 2,
        }


class IERDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "Give instructions to edit the image.",
            "How would you edit the image?",
            "What edits should I do no make the second image from the first?",
            "Image editing instructions:",
            "Image editing requests:",
        ]

        er_test = join(vis_root, "train.json")
        with open(er_test) as f:
            self.er_data = json.load(f)

    def __len__(self):
        return len(self.er_data)

    def __getitem__(self, index):
        sample = self.er_data[index]
        image_file1 = join(self.vis_root, "images", sample["img0"])
        image_file2 = join(self.vis_root, "images", sample["img1"])
        image1 = Image.open(image_file1).convert("RGB")
        image2 = Image.open(image_file2).convert("RGB")
        image1 = self.vis_processor(image1)
        image2 = self.vis_processor(image2)
        image = torch.stack([image1, image2], dim=0)

        caption = sample["sents"][random.randint(0, len(sample["sents"]) - 1)]
        caption = self.text_processor(caption)

        instruction = f"<Img><ImageHere></Img> <Img><ImageHere></Img> [idc] {random.choice(self.instruction_pool)} "
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "length": 2,
            "path": image_file1,
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
        # image = torch.stack([image, image], dim=0)

        caption = info["caption_str"]
        caption = self.text_processor(caption)
        instruction = (
            "<Img><ImageHere></Img> <Img><ImageHere></Img> [caption] {} ".format(
                random.choice(self.instruction_pool)
            )
        )
        instruction_one = "<Img><ImageHere></Img> [caption] {} ".format(
            random.choice(self.instruction_pool)
        )
        return {
            "image": image,
            "instruction_input": instruction_one,
            "answer": caption,
            # "length": 2,
        }

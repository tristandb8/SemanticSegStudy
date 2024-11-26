from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import os
import torch
from torch import nn
from torchvision import transforms
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from PIL import Image
from typing import Literal
from models.FPNEfficientNet import FPNEfficientNetV2_S
from models.FPNMobileNet import FPNMobileNetV3Large

def validate(model, dataloader, best_scores, epoch, device, model_name):
    model.eval()
    metric = evaluate.load("mean_iou")
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, masks, class_labels, _ in tepoch:
                images = images.to(device)
                masks = masks.to(device)
            
                if model_name == "mask2former_swin":
                    class_labels = class_labels.to(device)
                    outputs = model(pixel_values=images, mask_labels=masks, class_labels=class_labels)
                    predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512) for i in range(masks.shape[0])]))
                    masks = masks.argmax(dim=1)
                elif model_name == "segformer_mit":
                    outputs = model(pixel_values=images, labels=masks)
                    upsampled_logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)
                elif model_name == "fpn_efficientnet" or "fpn_mobilenet":
                    outputs = model(images)
                    upsampled_logits = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    loss = criterion(upsampled_logits, masks.long())
                    predicted = upsampled_logits.argmax(dim=1)
                else:
                    assert False

                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
            metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=masks.cpu(),
                    num_labels=num_classes,
                    ignore_index=255,
                    reduce_labels=False, # we've already reduced the labels ourselves
                )
            print("==========VAL=========")
            for score in ["iou", "accuracy"]:
                if metrics[f"mean_{score}"] > best_scores[score]["score"]:
                    best_scores[score]["score"] = metrics[f"mean_{score}"]
                    best_scores[score]["epoch"] = epoch
                    print("---NEW BEST---")
                print(f"Mean_{score}: {metrics[f'mean_{score}']} (best: {best_scores[score]})")



if __name__ == '__main__':

    # Create datasets and dataloaders
    base_folder = r"bcss_sample"
    model_name: Literal["segformer_mit", "mask2former_swin", "fpn_efficientnet", "fpn_mobilenet"] = "fpn_mobilenet"
    num_classes = 22
    num_epochs = 5
    bs = 16
    
    id2label = {0:"background"}
    for i in range(1, 22):
        id2label[i] = "class_" + str(i)

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_image_dir = os.path.join(base_folder, "train")
    train_mask_dir = os.path.join(base_folder, "train_mask")
    val_image_dir = os.path.join(base_folder, "val")
    val_mask_dir = os.path.join(base_folder, "val_mask")

    train_dataset = SegmentationDataset(train_image_dir,
                                        train_mask_dir,
                                        model_name,
                                        transform=transform,
                                        augment=True)
    val_dataset = SegmentationDataset(val_image_dir,
                                      val_mask_dir,
                                      model_name,
                                      transform=transform,
                                      augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda")
    if model_name == "mask2former_swin":
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic", id2label=id2label, ignore_mismatched_sizes=True).to(device)
    elif model_name == "segformer_mit":
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_classes).to(device)
    elif model_name == "fpn_efficientnet":
        model = FPNEfficientNetV2_S(num_classes=num_classes).cuda()
    elif model_name == "fpn_mobilenet":
        model = FPNMobileNetV3Large(num_classes=num_classes).cuda()
    else:
        assert False

    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    metric = evaluate.load("mean_iou")
    best_scores = {"iou": {"epoch": -1, "score": -1},
                   "accuracy": {"epoch": -1, "score": -1}}

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)
        with tqdm(train_dataloader, unit="batch") as tepoch:
            model.train()
            for images, masks, class_labels, _ in tepoch:
                optimizer.zero_grad()
                # get the inputs
                images = images.to(device)
                masks = masks.to(device)
                if model_name == "mask2former_swin":
                    class_labels = class_labels.to(device)
                    outputs = model(pixel_values=images, mask_labels=masks, class_labels=class_labels)
                    predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512) for i in range(masks.shape[0])]))
                    masks = masks.argmax(dim=1)
                    loss = outputs.loss
                elif model_name == "segformer_mit":
                    outputs = model(pixel_values=images, labels=masks)
                    upsampled_logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)
                    loss = outputs.loss
                elif model_name == "fpn_efficientnet" or "fpn_mobilenet":
                    outputs = model(images)
                    upsampled_logits = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    loss = criterion(upsampled_logits, masks.long())
                    predicted = upsampled_logits.argmax(dim=1)
                else:
                    assert False

                loss.backward()
                optimizer.step()

                # evaluate
                with torch.no_grad():
                    metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())

            # currently using _compute instead of compute
            # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
            metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=masks.cpu(),
                    num_labels=num_classes,
                    ignore_index=255,
                    reduce_labels=False, # we've already reduced the labels ourselves
                )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])

            validate(model, val_dataloader, best_scores, epoch + 1, device, model_name)
    print(best_scores)
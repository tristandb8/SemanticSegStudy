from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, AutoImageProcessor
import os
import torch
from torch import nn
from torchvision import transforms
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from typing import Literal
from models.CustomModels import CustomModel
import json
import shutil


def validate(model, dataloader, best_scores, epoch, device, architecture):
    model.eval()
    metric = evaluate.load("mean_iou")
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, masks in tepoch:
                images = images.to(device)
                masks = masks.to(device)
            
                if architecture == "segformer":
                    outputs = model(pixel_values=images, labels=masks)
                    predicted = outputs.logits.argmax(dim=1)
                else:
                    outputs = model(images)
                    predicted = outputs.argmax(dim=1)

                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
            metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=masks.cpu(),
                    num_labels=args["num_classes"],
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

    with open('args.json', 'r') as file:
        args = json.load(file)

    # Set up data transforms
    preprocessor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_image_dir = os.path.join(args["base_folder"], "train")
    train_mask_dir = os.path.join(args["base_folder"], "train_mask")
    val_image_dir = os.path.join(args["base_folder"], "val")
    val_mask_dir = os.path.join(args["base_folder"], "val_mask")

    train_dataset = SegmentationDataset(train_image_dir,
                                        train_mask_dir,
                                        preprocessor=preprocessor,
                                        augment=True)
    val_dataset = SegmentationDataset(val_image_dir,
                                      val_mask_dir,
                                      preprocessor=preprocessor,
                                      augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args["bs"], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args["bs"], shuffle=False, num_workers=8)
    
    final_data = {}
    
    model_configs = []
    for encoder in ["resnet34",
                    "mobilenetv3_large_100",
                    "mobilenetv4_conv_small.e2400_r224_in1k",
                    "mobilenetv4_hybrid_medium.ix_e550_r384_in1k",
                    "efficientnet_b0",
                    "rexnetr_200.sw_in12k_ft_in1k"]:
        for architecture in ["FPN", "Unet"]:
            model_configs.append([encoder, architecture])
    model_configs.append(["mit-b0", "segformer"])
    for encoder, architecture in model_configs:
        final_data[encoder] = {}
    for encoder, architecture in model_configs:
            # Initialize the model, loss function, and optimizer
            device = torch.device("cuda")
            if architecture == "segformer":
                model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=args["num_classes"]).to(device)
            else:
                model = CustomModel(num_classes=args["num_classes"], backbone=encoder, decoder=architecture).to(device)

            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=args["starting_lr"])

            metric = evaluate.load("mean_iou")
            best_scores = {"iou": {"epoch": -1, "score": -1}, "accuracy": {"epoch": -1, "score": -1}}
            last_lr_step = -1

            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(args["num_epochs"]):
                print("Epoch:", epoch + 1)
                with tqdm(train_dataloader) as tepoch:
                    model.train()
                    for images, masks in tepoch:
                        optimizer.zero_grad()
                        # get the inputs
                        images = images.to(device)
                        masks = masks.to(device)
                        if architecture == "segformer":
                            outputs = model(pixel_values=images, labels=masks)
                            predicted = outputs.logits.argmax(dim=1)
                            loss = outputs.loss
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                            predicted = outputs.argmax(dim=1)

                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())


                    metrics = metric._compute(
                            predictions=predicted.cpu(),
                            references=masks.cpu(),
                            num_labels=args["num_classes"],
                            ignore_index=255,
                            reduce_labels=False
                        )

                    print("Loss:", loss.item())
                    print("Mean_iou:", metrics["mean_iou"])
                    print("Mean accuracy:", metrics["mean_accuracy"])

                    validate(model, val_dataloader, best_scores, epoch + 1, device, architecture)

                    # Reduce LR if no improvement for 'args["patience"]' epochs
                    if epoch - max(best_scores['iou']['epoch'], best_scores['accuracy']['epoch'], last_lr_step) >= args["patience"]:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= args["lr_factor"]
                        last_lr_step = epoch
                        print(f"Reducing LR to {optimizer.param_groups[0]['lr']:.9f}")

            print("_______________________FINAL SCORE__________________________")
            print(architecture, encoder, best_scores)
            final_data[encoder][architecture] = best_scores
            print("____________________________________________________________")

    save_dir = f"experiments/{args['exp_name']}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy("args.json", f"{save_dir}/args.json")
    with open(f"{save_dir}/data.json", "w") as f:
        json.dump(final_data, f, indent=4)
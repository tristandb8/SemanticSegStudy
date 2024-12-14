from transformers import SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor
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

def validate(model, dataloader, best_scores, epoch, device, architecture):
    model.eval()
    metric = evaluate.load("mean_iou")
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, masks, class_labels, _ in tepoch:
                images = images.to(device)
                masks = masks.to(device)
            
                if architecture == "mask2former":
                    class_labels = class_labels.to(device)
                    outputs = model(pixel_values=images, mask_labels=masks, class_labels=class_labels)
                    predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(masks.shape[-1], masks.shape[-2]) for i in range(masks.shape[0])]))
                    masks = masks.argmax(dim=1)
                    print(masks[0, :8, :8], predicted[0, :8, :8])
                elif architecture == "segformer":
                    outputs = model(pixel_values=images, labels=masks)
                    predicted = outputs.logits.argmax(dim=1)
                else:
                    outputs = model(images)
                    predicted = outputs.argmax(dim=1)

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
    exp_name = "test"
    num_classes = 22
    num_epochs = 10
    bs = 16
    patience = 2
    lr_factor = 0.2
    one_hot_masks = True

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
                                        one_hot=one_hot_masks,
                                        transform=transform,
                                        augment=True)
    val_dataset = SegmentationDataset(val_image_dir,
                                      val_mask_dir,
                                      one_hot=one_hot_masks,
                                      transform=transform,
                                      augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
    
    final_data = {}
    
    model_configs = []
    # for encoder in ["resnet34",
    #                 "mobilenetv3_large_100",
    #                 "mobilenetv4_conv_small.e2400_r224_in1k",
    #                 "mobilenetv4_hybrid_medium.ix_e550_r384_in1k",
    #                 "efficientnet_b0",
    #                 "rexnetr_200.sw_in12k_ft_in1k",
    #                 "maxvit_base_tf_512.in21k_ft_in1k"]:
    #     for architecture in ["FPN", "Unet"]:
    #         model_configs.append([encoder, architecture])
    #         break
    #     break
    model_configs.append(["swin-t", "mask2former"])
    # model_configs.append(["mit-b0", "segformer"])
    for encoder, architecture in model_configs:
        final_data[encoder] = {}
    for encoder, architecture in model_configs:
            # Initialize the model, loss function, and optimizer
            device = torch.device("cuda")
            if architecture == "mask2former":
                assert one_hot_masks
                id2label = {0:"background"}
                for i in range(1, 22):
                    id2label[i] = "class_" + str(i)
                preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
                model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic", id2label=id2label, ignore_mismatched_sizes=True).to(device)
            elif architecture == "segformer":
                model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_classes).to(device)
            else:
                model = CustomModel(num_classes=22, backbone=encoder, decoder=architecture).to(device)


            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            metric = evaluate.load("mean_iou")
            best_scores = {"iou": {"epoch": -1, "score": -1}, "accuracy": {"epoch": -1, "score": -1}}

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
                        if architecture == "mask2former":
                            class_labels = class_labels.to(device)
                            outputs = model(pixel_values=images, mask_labels=masks, class_labels=class_labels)
                            predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(masks.shape[-1], masks.shape[-2]) for i in range(masks.shape[0])]))
                            masks = masks.argmax(dim=1)
                            print(masks[0, :8, :8], predicted[0, :8, :8])
                            loss = outputs.loss
                        elif architecture == "segformer":
                            outputs = model(pixel_values=images, labels=masks)
                            predicted = outputs.logits.argmax(dim=1)
                            loss = outputs.loss
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                            predicted = outputs.argmax(dim=1)

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

                    validate(model, val_dataloader, best_scores, epoch + 1, device, architecture)


                    if epoch - max(best_scores['iou']['epoch'], best_scores['accuracy']['epoch']) >= patience:  # Reduce LR if no improvement for 'patience' epochs
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_factor
                        print(f"Reducing LR to {optimizer.param_groups[0]['lr']:.9f}")

            print("______________________________________________________")
            print("____________________FINAL SCORE_______________________")
            print("______________________________________________________")
            print(architecture, encoder, best_scores)
            final_data[encoder][architecture] = best_scores
            print("______________________________________________________")
            print("______________________________________________________")
            print("______________________________________________________")
    
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    with open(f"experiments/scores_{exp_name}.json", "w") as f:
        json.dump(final_data, f, indent=4)
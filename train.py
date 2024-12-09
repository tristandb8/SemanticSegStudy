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
                    predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512) for i in range(masks.shape[0])]))
                    masks = masks.argmax(dim=1)
                elif architecture == "segformer":
                    outputs = model(pixel_values=images, labels=masks)
                    upsampled_logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)
                else:
                    outputs = model(images)
                    upsampled_logits = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)


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
    encoder: Literal["resnet34",
                     "mobilenetv3_large_100",
                     "mobilenetv4_conv_small.e2400_r224_in1k",
                     "mobilenetv4_hybrid_medium.ix_e550_r384_in1k",
                     "efficientnet_b0",
                     "rexnetr_200.sw_in12k_ft_in1k",
                     "maxvit_base_tf_512.in21k_ft_in1k"] = "mobilenetv4_conv_small.e2400_r224_in1k"
    num_classes = 22
    num_epochs = 5
    bs = 4
    patience = 2
    lr_factor = 0.2
    one_hot_masks = False

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

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=1)

    for architecture in ["FPN, ""Unet"]:
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
        
        print("_____TEST______")
        x = torch.randn(4, 3, 512, 512).to(device)
        print(f"Input shape: {x.shape}")
        output = model(x)
        print(f"Output shape: {output.shape}")
        print("_______________")

        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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
                        predicted = torch.stack(preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512) for i in range(masks.shape[0])]))
                        masks = masks.argmax(dim=1)
                        loss = outputs.loss
                    elif architecture == "segformer":
                        outputs = model(pixel_values=images, labels=masks)
                        upsampled_logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                        predicted = upsampled_logits.argmax(dim=1)
                        loss = outputs.loss
                    else:
                        outputs = model(images)
                        upsampled_logits = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                        loss = criterion(upsampled_logits, masks.long())
                        predicted = upsampled_logits.argmax(dim=1)

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
                    print(f"Reducing LR to {optimizer.param_groups[0]['lr']:.6f}")

        print("______________________________________________________")
        print("____________________FINAL SCORE_______________________")
        print("______________________________________________________")
        print(architecture, encoder, best_scores)
        print("______________________________________________________")
        print("______________________________________________________")
        print("______________________________________________________")

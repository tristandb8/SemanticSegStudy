# Simple semantic segmentation experiment

Compare default backbone performances

### Attention!

Lack of time to spend in this project, many flaws:
- This dataset has heavy class imbalanced, this has not been addressed
- Performing poorer at epoch 20, doesn't mean performing poorly on epoch 100. Lack of resources will pause this for now
- PLUS desired constraints: 1/4 output size and mask


## Results

| Encoder | Architecture | IoU | Accuracy | Time (ms) |
|---------|--------------|-----|----------|----------|
| resnet34  | **FPN**          | 28.7 | 43.2 | **6.8** |
| resnet34  | Unet         | 26.7 | 39.4 | 7.5 |
| mobilenetv3_large_100 | **FPN** | 27.9 | 44.0 | 10.2 |
| mobilenetv3_large_100 | Unet | 24.5 | 36.9 | 9.5 |
| mobilenetv4_conv_small.e2400_r224_in1k | **FPN** | 29.8 | 41.1 | 8.2 |
| mobilenetv4_conv_small.e2400_r224_in1k | Unet | 25.9 | 35.8 | 8.6 |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | **FPN** | **36.0** | **49.5** | 21.0 |
| mobilenetv4_hybrid_medium.ix_e550_r384_in1k | Unet | 31.4 | 46.6 | 20.3 |
| efficientnet_b0 | **FPN** | 26.6 | 45.5 | 12.2 |
| efficientnet_b0 | Unet | 27.2 | 37.5 | 12.7 |
| rexnetr_200.sw_in12k_ft_in1k | FPN | 30.9 | 45.4 | 13.1 |
| rexnetr_200.sw_in12k_ft_in1k | **Unet** | 32.3 | 46.3 | 13.9 |
| mit-b0 | segformer | 31.0 | 45.6 | 10.9 |

## Early! results


Best performance: **mobilenetv4_hybrid_medium (FPN)**

Best performance/speed:  **rexnetr_200 (Unet)**

## License

MIT License

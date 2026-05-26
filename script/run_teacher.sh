
#!/bin/bash
cd src
python -m training.main \
    --train-data "csvfiles/train/captions_train_new-2.csv"\
    --val-data="csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --save-frequency 0 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 150 \
    --batch-size 8 \
    --lr 1e-05 \
    --wd 0.1 \
    --epochs 15 \
    --workers=1 \
    --seed 42 \
    --logs "../logs" \
    --name "ViT_B_16-laion400m_teacher-marine-old-26"

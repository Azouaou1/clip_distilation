cd src
python -m training.main \
    --val-data="csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --eval \
    --workers=1 \
    --seed 42 \
    --logs "../logs" \
    --name "eval-pretrained-vitb16-2"
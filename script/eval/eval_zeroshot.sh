#!/bin/bash
cd ..

# Comparaison zero-shot teacher (ViT-B-16) vs student (ViT-T-16)
# Le teacher represente 100% de reference — on mesure ce que le student a retenu
python zero-shot/evaluation_zero_shot.py \
    --model            ViT-T-16 \
    --model-checkpoint /chemin/vers/logs/ViT_T_16_student_kd/checkpoints/epoch_latest.pt \
    --t-model            ViT-B-16 \
    --t-model-checkpoint /chemin/vers/logs/ViT_B_16_teacher/checkpoints/epoch_latest.pt \
    --dataset    src/csvfiles/test/captions_morphology_test_clip.csv \
    --data-root  /chemin/vers/tes/images/ \
    --label-column phylum \
    --batch-size 32 \
    --workers    4 \
    --seed       42

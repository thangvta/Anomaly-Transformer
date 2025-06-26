export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path dataset/HL19 --input_c 20    --output_c 20
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path dataset/HL19  --input_c 20    --output_c 20  --pretrained_model 20



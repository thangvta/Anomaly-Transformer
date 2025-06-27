export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset HL19  --data_path dataset/HL19 --input_c 19    --output_c 19
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset HL19   --data_path dataset/HL19  --input_c 19   --output_c 19  --pretrained_model 20



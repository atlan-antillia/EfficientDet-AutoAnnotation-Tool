rem 1.3_pretrain.bat
python ../../ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./pretrain_tfrecords/train/*.tfrecord  ^
  --val_file_pattern=./pretrain_tfrecords/valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="input_rand_hflip=False,image_size=512x512,num_classes=5,label_map=./label_map.yaml" ^
  --model_dir=./pretrained_models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./pretrain_eval ^
  --ckpt=../../efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=1 ^
  --eval_samples=400  ^
  --num_examples_per_epoch=800 ^
  --num_epochs=100

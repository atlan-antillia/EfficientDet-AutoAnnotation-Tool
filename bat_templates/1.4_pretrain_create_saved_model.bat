rem 1.4_pretrain_create_saved_model.bat
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./pretrained_models  ^
  --hparams="image_size={IMAGE_SIZE},num_classes={NUM_CLASSES}" ^
  --saved_model_dir=./pretrained_saved_model

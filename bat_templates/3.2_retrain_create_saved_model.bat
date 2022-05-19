rem 3.2_retrain_create_saved_model.bat
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./retrained_models  ^
  --hparams="image_size={IMAGE_SIZE},num_classes={NUM_CLASSES}" ^
  --saved_model_dir=./retrained_saved_model

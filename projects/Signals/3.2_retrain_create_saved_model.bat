rem 3.2_retrain_create_saved_model.bat
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./retrained_models  ^
  --hparams="image_size=512x512,num_classes=5" ^
  --saved_model_dir=./retrained_saved_model

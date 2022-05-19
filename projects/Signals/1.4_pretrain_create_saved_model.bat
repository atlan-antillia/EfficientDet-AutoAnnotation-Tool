rem 1.4_pretrain_create_saved_model.bat
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./pretrained_models  ^
  --hparams="image_size=512x512,num_classes=5" ^
  --saved_model_dir=./pretrained_saved_model

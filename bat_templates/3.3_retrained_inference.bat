rem 3.3_retrained_inference.bat
python ../../SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./retrained_saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="num_classes={NUM_CLASSES},label_map=./label_map.yaml" ^
  --input_image=./test_dataset/*.jpg ^
  --output_image_dir=./test_dataset_outputs

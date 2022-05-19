rem 2.2_apply_autoannotator.bat
python ../../AutoAnnotator.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./pretrained_saved_model ^
  --min_score_thresh=0.6 ^
  --hparams="num_classes={NUM_CLASSES},label_map=./label_map.yaml" ^
  --input_image=./unannotated_dataset/*.jpg ^
  --yolo_output_dir=./autoannotated_yolo ^
  --yolo_classes_file=./classes.txt ^
  --yolo_autoannotation=True ^
  --output_image_dir=./unannotated_dataset_output

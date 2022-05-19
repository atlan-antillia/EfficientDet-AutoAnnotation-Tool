rem 1_create_pretrained_model.bat

call 1.1_pretrain_yolo_master_splitter.bat
call 1.2_pretrain_yolo2tfrecord_converter.bat
call 1.3_pretrain.bat
call 1.4_pretrain_create_saved_model.bat

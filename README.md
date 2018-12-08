# Kaggle_airbus_detection_detectron_pytorch and unet training edition

===========================================================================================================================
## [My teamleader's program](https://github.com/pascal1129/kaggle_airbus_ship_detection)  is based on https://github.com/facebookresearch/Detectron ,since I have no environment of Caffe2,I reimplement my teamleader's program in pytorch-detectron .This repository is just for record some change. The perfect edition is in [My teamleader's program](https://github.com/pascal1129/kaggle_airbus_ship_detection) ,and this repository get the 21st in [Kaggle_airbus_ship_detection](https://www.kaggle.com/c/airbus-ship-detection)
---------------------------------------------------------------------------------------------------------------------------
#### 1.pytorch_detectron link is https://github.com/roytseng-tw/Detectron.pytorch .Please refer to this link for detailed environmental configuration.

#### 2.data_preparation: the codes using for making data from https://github.com/pascal1129/kaggle_airbus_ship_detection 0_rle_to_coco/ .In this repository ,I put this code in the cocodataset_making/. Before using the detectron_pytorch to train the dataset, I made the data further into the format of COCO2017.If you want to use your own custom data format ,you need to add code in  dataset_catalog.py like my team leader's code do.
'''Python
'airbus_2018_train': {
        _IM_DIR:
            '/data/airbus_dataset/input/ships_train2018',
        _ANN_FN:
            '/data/airbus_dataset/input/annotations/instances_ships_train2018.json'
    },
'''
#### 3.train the dataset: I use the e2e_mask_rcnn_R-101-FPN_1x.yaml in the config/baselines/ ,here is some parameters is changed 1.NUM_CLASSES changed to 1    2.BASE_LR: 0.02  3. MAX_ITER: 20000 STEPS: [0, 5000, 10000] the origin number is too large lead to slow training.  The dummy_datasets.py is need to be changed. I use infer_simple_mask2csv.py to replace the origin infer_simple.py to get the submmsion.csv directly .I have a very detailed comment in the code.

#### 4. using unet to train the dataset for ensemble the result based on vote method to reduce the false positive: the unet using here is from https://www.kaggle.com/hmendonca/u-net-model-with-submission
## I retrain the unet locally and make it as unet_keras.py format without some Visualization method .

#### 5. submit the ensemble result :use the get_final_csv.py to ensemble the detectron's result and unet's result and add the no-ship infomation.


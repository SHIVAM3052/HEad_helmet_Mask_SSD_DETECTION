# HEad_helmet_Mask_SSD_DETECTION

Step 1 : Read the dataset and annotation file

Step 2 : Convert XML file into Dataframe and took out Head as labels, BBox, and attributes(helmet,no-helmet,mask,no-mask). Using this dataframe generate tf.recorsd for testing and training Check XML-2-tfrecords.py

Step 3 : Use TensorFlow Object Detection method to predict labels, BBox on our Dataset

Step 4 : First, Trained on Pre-trained model(Tensorflow Zoo) SSD_RESNET_50. Check head_detect_ssd_resnet_50.ipynb
        Results : 1.Training Time is more. It took 6 hrs  train my model on colab
                    2.Detction threshold kept quite low
                    3.Not able to detect biunding box on low resolution images.

Step 5 : Second, Trained on Pre-trained model(Tensorflow Zoo) SSD_EFFDETD0. Check head_detect_ssd_effnetd07.ipynb
        Results : 1.Training Time is less. It took 2 hrs  train my model on colab
                  2.Detection threshold optimum
                  3.Able to detect biunding box on low resolution images.

Step 6 : Models shared on this link for evaluation :=

**Issues:**
1. Don't have my own GPU. So trained on Google colab. Model take lots of time to train and my GPU quota got exhausted on a daily basis. Not able to optimize model more robust.
2. Data Inconsistency in Dataset: Images of multiple sizes were present. So, While training the model detailed imformation like feature extraction is not that much good. SO thresholding of BBox kept low for detection.
3. Time limit to complete two task. 

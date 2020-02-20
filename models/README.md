# Models
Dowload and unzip [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz).This unzipped will give you a folder with frozen weights proto buffer (which is the base format to optimize or convert to FP16, INT8, formats ) as well as a saved_model folder which contains the model and weights in TF Serving compatible model- basically, the folder has a saved_model.pb and optionally a folder containing variables. Since TF Serving has a concept of versioning, it will check for numerical folders under the base folder to serve. You can provide a configuration file giving specific version also which I will show below. I am so renaming the saved_folder to ‘01’ so that TF Serving can find it.

```shell
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -zxvf ssd_inception_v2_coco_2018_01_28.tar.gz
mv ssd_inception_v2_coco_2018_01_28/saved_model/ ssd_inception_v2_coco_2018_01_28/01/
rm ssd_inception_v2_coco_2018_01_28.tar.gz
```

Use saved_model_cli to get the signature_def of this model.

```shell
saved_model_cli show --all --dir ./ssd_inception_v2_coco_2018_01_28/01/
```

The output of saved_model_cli gives you how to request and parse the output.

>MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
>
>signature_def['serving_default']:
>  The given SavedModel SignatureDef contains the following input(s):
>    inputs['inputs'] tensor_info:
>        dtype: DT_UINT8
>        shape: (-1, -1, -1, 3)
>        name: image_tensor:0
>  The given SavedModel SignatureDef contains the following output(s):
>    outputs['detection_boxes'] tensor_info:
>        dtype: DT_FLOAT
>        shape: (-1, 100, 4)
>        name: detection_boxes:0
>    outputs['detection_classes'] tensor_info:
>        dtype: DT_FLOAT
>        shape: (-1, 100)
>        name: detection_classes:0
>    outputs['detection_scores'] tensor_info:
>        dtype: DT_FLOAT
>        shape: (-1, 100)
>        name: detection_scores:0
>    outputs['num_detections'] tensor_info:
>        dtype: DT_FLOAT
>        shape: (-1)
>        name: num_detections:0
>  Method name is: tensorflow/serving/predict
# Models
Dowload and unzip [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz).This unzipped will give you a folder with frozen weights proto buffer (which is the base format to optimize or convert to FP16, INT8, formats ) as well as a saved_model folder which contains the model and weights in TF Serving compatible model- basically, the folder has a saved_model.pb and optionally a folder containing variables. Since TF Serving has a concept of versioning, it will check for numerical folders under the base folder to serve. You can provide a configuration file giving specific version also which I will show below. I am so renaming the saved_folder to ‘01’ so that TF Serving can find it.

```shell
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -zxvf ssd_inception_v2_coco_2018_01_28.tar.gz
rm ssd_inception_v2_coco_2018_01_28.tar.gz
mv saved_model/ 01/
```





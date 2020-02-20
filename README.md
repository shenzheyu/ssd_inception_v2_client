# Client for ssd_inception_v2

- Creat vitual environment and install packages

```shell
python3 -m venv venv
source ./venv/bin/activate
pip install tensorflow==1.14.0 tensorflow-serving-api==1.14.0 opencv-python
deactivate
```

- Serve model via TF Serving

```shell
docker run -it --rm -p 8900:8500 --runtime=nvidia -v /home/shenz/ssd_inception_v2_client/models/ssd_inception_v2_coco_2018_01_28:/models/ssd_inception_v2_coco -e MODEL_NAME=ssd_inception_v2_coco tensorflow/serving:latest-gpu --enable_batching=true
```

- Use client.py to do request

```shell
python client.py -num_tests=1 -server=127.0.0.1:8900 -batch_size=1 -img_path='./example/000000000001.jpg'
```

- Use batch_test.py to test different batching size influences inference throughput.

```shell
python batch_test.py -server=127.0.0.1:8900
```


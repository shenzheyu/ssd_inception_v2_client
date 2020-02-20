from __future__ import division

import grpc
import tensorflow as tf
import time
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from image_pre_processing import decode_image_opencv

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.app.flags.DEFINE_string('img_path', '', 'realtive/fullpath to jpegfile')
tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
_draw = None
_label_map = None


_response_awaiting = True


def batch_inference(server, img_path):
    for batch_size_index in range(0, 6):
        batch_size = 2 ** batch_size_index
        print("Batch size = %d: (sec per image)" % batch_size)
        mean = 0
        first = 0
        second = 0
        for test_index in range(30):
            time = do_inference(server, batch_size, img_path) / batch_size
            mean += time
            print("    test %d: %f" % (test_index + 1, time))
            if test_index == 0:
                first = time
            elif test_index == 1:
                second = time
        print("    mean: %f" % (mean / 30))
        print("    first variance: %f" % ((first - second) * batch_size))


def do_inference(server, batch_size, img_path):
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ssd_inception_v2_coco'
    request.model_spec.signature_name = 'serving_default'
    IMAGENET_MEAN = (103.939, 116.779, 123.68)
    image, org = decode_image_opencv(img_path, max_height=800, swapRB=True, imagenet_mean=IMAGENET_MEAN)
    image = image.astype(np.uint8)
    start = time.time()
    input = image
    inputs = input
    for _ in range(batch_size - 1):
        inputs = np.append(inputs, input, axis=0)
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))
    stub.Predict(request, 60)
    end = time.time()
    return end - start


def main(_):
    if not FLAGS.img_path:
        print('Please specify img_path -img_path=...')
        return
    if not FLAGS.server:
        print('please specify server -server host:port')
        return
    batch_inference(FLAGS.server, FLAGS.img_path)


if __name__ == '__main__':
    print("SSD TFServing Client  <-server=127.0.0.1:8500>")
    print("Override this default value by command line args")
    tf.app.run()
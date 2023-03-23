from tensorflow.python.client import device_lib
import sys
import tensorflow as tf


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Python {sys.version}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

print("-----------")

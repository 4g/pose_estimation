import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow import keras
from tensorflow import lite
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import segmentation_models as sm
from . import  tflitemodel

def get_model(model_path):
    model = keras.models.load_model(model_path)
    # model = keras.applications.mobilenet.MobileNet(alpha=0.25, include_top=False, input_shape=(224,224,3))
    return model

def model_compression_test(model, image, out):
    # model = tfmot.quantization.keras.quantize_model(model)
    model.summary()

    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = False
    tflite_model = converter.convert()

    out_path = 'model.tflite'
    if out:
        out_path = out

    with tf.io.gfile.GFile(out_path, 'wb') as f:
        f.write(tflite_model)

    mdl = tflitemodel.TFLiteModel().load_model(out_path)
    mdl.speedtest(image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="pass a sample image", required=True)
    parser.add_argument("--model", help="pass a model file", required=False, default=None)
    parser.add_argument("--out", help="tflite output path", required=False, default=None)

    args = parser.parse_args()
    model = get_model(args.model)
    model_compression_test(model, args.image, args.out)

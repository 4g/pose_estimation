import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

def create_pose_model(img_width, img_height, num_keypoints):
    backbone_model = keras.applications.mobilenet.MobileNet
    backbone = backbone_model(input_shape=(img_width, img_height, 3),
                              include_top=False,
                              alpha=0.25)

    keypoints_conv = keras.layers.Conv2D(filters=num_keypoints,
                                         kernel_size=1,
                                         strides=1,
                                         activation='sigmoid',
                                         use_bias=True)

    offsets_conv = keras.layers.Conv2D(filters=num_keypoints*2,
                                        kernel_size=1,
                                        strides=1,
                                        use_bias=False)

    offsets_conv_out = offsets_conv(backbone.output)
    offsets_conv_activated = tf.keras.layers.ReLU(max_value=1.0, negative_slope=0, threshold=0)(offsets_conv_out)

    out1 = keypoints_conv(backbone.output)
    out2 = offsets_conv_activated
    concat = keras.layers.concatenate([out1, out2], axis=-1)

    model = keras.models.Model(inputs=backbone.input, outputs=[concat])
    return model


def pose_model(img_width, img_height, num_keypoints):
    unet = sm.Unet(backbone_name='mobilenet',
                   input_shape=(img_width, img_height, 3),
                   encoder_weights='imagenet',
                   decoder_filters=(),
                   classes=num_keypoints * 3,
                   alpha=0.75)
    return unet

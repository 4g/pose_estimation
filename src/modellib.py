import segmentation_models as sm

def create_pose_model(img_width, img_height, num_keypoints):
    return pose_model(img_width, img_height, num_keypoints)
    # backbone = backbone_model(input_shape=(img_width, img_height, 3),
    #                           include_top=False,
    #                           alpha=.5)
    #
    # # for layer in backbone.layers:
    # #     if layer.__class__.__name__ == 'BatchNormalization':
    # #         layer.trainable = False
    #
    # keypoints_conv = keras.layers.Conv2D(filters=num_keypoints,
    #                                      kernel_size=1,
    #                                      strides=1,
    #                                      activation='sigmoid',
    #                                      use_bias=True)
    #
    # offsets_conv = keras.layers.Conv2D(filters=num_keypoints*2,
    #                                     kernel_size=1,
    #                                     strides=1,
    #                                     activation='sigmoid',
    #                                     use_bias=False)
    #
    # offsets_conv_out = offsets_conv(backbone.output)
    #
    # out1 = keypoints_conv(backbone.output)
    # out2 = offsets_conv_out
    # concat = keras.layers.concatenate([out1, out2], axis=-1)
    #
    # model = keras.models.Model(inputs=backbone.input, outputs=[concat])
    # return model


def pose_model(img_width, img_height, num_keypoints):
    unet = sm.Unet(backbone_name='mobilenetv2',
                   input_shape=(img_width, img_height, 3),
                   encoder_weights='imagenet',
                   decoder_filters=(32, 8, 8),
                   classes=num_keypoints,
                   alpha=0.35)
    return unet

if __name__ == '__main__':
    model = pose_model(256, 256, 13)
    print(model.summary())
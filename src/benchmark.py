from tensorflow import  keras
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import segmentation_models as sm

policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

mnv2 = sm.Unet(backbone_name='mobilenet',
                   input_shape=(192, 192, 3),
                   encoder_weights='imagenet',
                   decoder_filters=(64, 16, 16),
                   classes=13,
                   alpha=0.5)

test_images = np.zeros((10000, 192, 192, 3), dtype=np.uint8)
mnv2.compile(optimizer='adam', loss='mse', metrics=['mse'])

print("Prediction benchmark")
results = mnv2.predict(test_images, batch_size=32, verbose=1)

print("Training benchmark")
mnv2.fit(test_images, results, batch_size=64, epochs=5000, use_multiprocessing=False)

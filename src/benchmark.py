from tensorflow import  keras
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

mnv2 = keras.applications.mobilenet.MobileNet(alpha=0.5, include_top=False)
test_images = np.zeros((2000, 480, 480, 3), dtype=np.float32)
mnv2.compile(optimizer='adam', loss='mse', metrics=['mse'])

print("Prediction benchmark")
results = mnv2.predict(test_images, batch_size=32, verbose=1)

print("Training benchmark")
mnv2.fit(test_images, results, batch_size=8, epochs=5)
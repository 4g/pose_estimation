import tensorflow as tf
import h5py

# Load TFLite model and allocate tensors.
import sys
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get details for each layer
all_layers_details = interpreter.get_tensor_details()

f = h5py.File("mobilenet_v3_weights_infos.hdf5", "w")

for layer in all_layers_details:
    # to create a group in an hdf5 file
    grp = f.create_group(str(layer['index']))

    # to store layer's metadata in group's metadata
    grp.attrs["name"] = layer['name']
    grp.attrs["shape"] = layer['shape']
    # grp.attrs["dtype"] = all_layers_details[i]['dtype']
    grp.attrs["quantization"] = layer['quantization']

    weights = interpreter.get_tensor(layer['index'])

    print(layer['name'], layer['shape'], weights.shape, layer['quantization'])

    # to store the weights in a dataset
    grp.create_dataset("weights", data=interpreter.get_tensor(layer['index']))

f.close()

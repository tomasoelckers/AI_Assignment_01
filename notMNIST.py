import numpy as np
import gzip


# Extraction of data from file type .gz
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data


# Extraction of data
train_data = extract_data('Data/train-images-idx3-ubyte.gz', 60000)
test_data = extract_data('Data/t10k-images-idx3-ubyte.gz', 10000)

print(train_data[0].shape)

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_data.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_data.shape))

from pathlib import Path
import struct
from PIL import Image
import numpy as np
import struct

"""What I learned
1. The buffer's size in bytes must be a multiple of the size required by the format
2. If I want to get a value from a 32 bit integer, given by 4 bytes 0xabcdefgh, 
    I need to calculate it like follows: (a*16+b)*16**6 + (c*16+d)*16**4 + ...
    Because each byte contains 8 bits, which has 256 possible values, i.e. 2**8 = 16**2.
3. struct is very interesting. First I need to open the file for reading using the regular
    idiom. Then I need to read the right amount of bytes so that the buffer doesn't get angry

"""

#project_path = Path(__file__).parent.parent
MNIST_path = Path(r"E:\Data\Neural Networks\CNN\MNIST\raw")
MNIST_datasets_paths = list(MNIST_path.glob("*"))
training_images_path = MNIST_datasets_paths[4]

print(training_images_path)

# # 1. Read the magic number, 2051
# with open(training_images_path, 'rb') as file:
#     data = file.read(4)
#     data = struct.unpack('>BBBB', data)

# print(data[3]+data[2]*2**8)

# # 2. Get the number of items, 60,000
# with open(training_images_path, 'rb') as file:
#     data = file.read(8)
#     data = struct.unpack_from('>BBBB', data, offset = 4)

# print(data[3]*2**0+data[2]*2**8)

# # 3. Get the number of rows, 28
# with open(training_images_path, 'rb') as file:
#     data = file.read(12)
#     data = struct.unpack_from('>BBBB', data, offset = 8)

# print(data[3]*2**0+data[2]*2**8)

# # 4. Get the number of columns, 28
# with open(training_images_path, 'rb') as file:
#     data = file.read(16)
#     data = struct.unpack_from('>BBBB', data, offset = 12)

# print(data[3]*2**0+data[2]*2**8)

# #Get pixel values. Each value is just an unsigned char.
# with open(training_images_path, 'rb') as file:
#     data = file.read()
#     data = struct.iter_unpack('>B', data)

#img_data = list(data)[16:16+28**2]

#img = Image.frombytes('L', size = (28,28), data = img_data)

#TODO
# 1. examine an image
# 2. test the fastest way to get the images using this struct package.

# with open(training_images_path, 'rb') as file:
#     data = file.read(16+28**2)
# print(data[:16].hex('x'))
# img_data = data[16:]
# img = Image.frombytes('L', size = (28,28), data = img_data)
# img.show()

#source: https://stackoverflow.com/a/53181925/6745557
with open(training_images_path,'rb') as file:
    magic, size = struct.unpack(">II", file.read(8))
    nrows, ncols = struct.unpack(">II", file.read(8))
    #data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = np.fromfile(file, dtype=np.dtype('B'))

#What I learned:
# 1. I can ingest data directly into a numpy array using fromfile.
# 2. The newbyteorder('>') is not needed when using 'B' as dtype.
# 3. reshaping fills the array from rightmost dimension to leftmost dimension. The initial reshaping had to correspond to the way the data is stored in the file.
# 4. To get this data to look like a collage/tiling, i needed 4 dimensions for the first shaping, as my tiling would be a 244 by 244 grid of 28 by 28 images.
# 5. Then I needed to transpose the array to interleave the dimensions, so that it would be in the right order for another reshaping.
    # In this case, I transposed it to (100, 28, 100, 28), which means that when reshaping to 2800x*2800*, the second 2800 would consist of the right hand 100*28, i.e. the topmost row of the collage.
    # This was confirmed with Image.fromarray(data_collage[0:14:]).show() which shows half the topmost row of the collage.

#I can't take all the data because it doisn't reshape properly as 60,000 isn't a square number, and 244*244 is the largest square number below 60,000.
data_collage = data[:244**2*28**2]
data_collage = data_collage.reshape(244, 244, 28, 28)
data_collage = data_collage.transpose(0, 2, 1, 3).reshape(244*28, 244*28)
Image.fromarray(data_collage).show()

# data = data.reshape(size, nrows, ncols)
# Image.fromarray(data[0]).show()

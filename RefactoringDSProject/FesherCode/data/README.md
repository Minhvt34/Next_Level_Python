header\
size in dimension 0\
size in dimension 1\
size in dimension 2\
...\
size in dimension N\
data

the magic number is an integer (MSB first). The first 2 bytes are always 0.

the third byte codes the type of the data:\
0x08: usigned byte\
0x09: signed byte\
0x0B: short (2 bytes)\
0x0C: int (4 bytes)\
0x0D: float (4 bytes)\
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors. 2 for matrices...
The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array. i.e. the index in the last dimension changes the fastest.

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):\
[offset] [type]          [value]          [description]\
0000     32 bit integer  0x00000801(2049) magic number (MSB first)\
0004     32 bit integer  10000            number of items\
0008     unsigned byte   ??               label\
0009     unsigned byte   ??               label\
.........\
xxxx     unsigned byte   ??               label\
the labels values are 0 to 9.
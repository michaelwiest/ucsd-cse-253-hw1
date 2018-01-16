import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy


def mnist_printer(mnist_array, save=False):
    pixmap = weights_to_2d(mnist_array).astype(float)
    plt.imshow(pixmap, cmap=cm.gray, interpolation='nearest')
    plt.show()
    if save:
        img.save('somepic.png')

def weights_to_2d(weights):
    dim1 = int(np.sqrt(len(weights)))
    dim2 = int(len(weights) / dim1)
    weights = weights[:dim1*dim2] # This is for adding the occlusions.
    return copy.deepcopy(np.reshape(weights, (dim1, dim2)))

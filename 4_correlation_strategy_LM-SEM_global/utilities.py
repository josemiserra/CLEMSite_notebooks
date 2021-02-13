import matplotlib.pyplot as plt
import numpy as np

def show_images(images, cols = 1, titles = None, axes_empty = True, shrink = 1.0):
    """
    Display a list of images.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    cols: number of columns
    titles: List of titles corresponding to each image. 
    axes_empty: remove axis values (not visible) for images

    """
    n_images = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        if axes_empty:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.set_size_inches(np.array(fig.get_size_inches()*shrink) * n_images)
    plt.show()
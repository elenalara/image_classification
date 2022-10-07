import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import product

# Auxiliar functions for Tensorboard

def plot_each(num_each):
    fig = plt.figure()
    plt.bar(num_each.keys(), num_each.values())
    plt.xlabel("Classes names")
    plt.title("Label counts", bbox = {'facecolor':'0.8', 'pad':5}, fontsize = 13)
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
        #colapsa la imagen en la primera dimension (blanco y negro 1d, color sería 3d)
    img = img / 2 + 0.5 # unnormalize
    npimg = img.cpu().numpy()
    if one_channel: # plot en blanco y negro
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_grid(images):
    fig = plt.figure(figsize=(10, 10))
    for idx in range(9):
        ax = fig.add_subplot(3, 3, idx+1)
        matplotlib_imshow(images[idx], one_channel=True)
    return fig
        
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained network and a list of images
    '''
    probs = net(images).cpu()
    # convert output probabilities to predicted class
    preds_tensor = probs.argmax(dim=1)
    preds = np.squeeze(preds_tensor.numpy())
    #return preds, [act_transform(el)[i] for i, el in zip(preds, probs)]
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, probs)]

def plot_classes_preds(classes, net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images and labels from a batch, that shows
    the network's top prediction along with its probability, alongside the actual label, coloring this information
    based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(classes[preds[idx]], probs[idx] * 100.0,
            classes[labels[idx]]), color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def plot_image(i, classes, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array.cpu(), true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    matplotlib_imshow(img, one_channel=True)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label.cpu():
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f"{classes[predicted_label]} {100 * max(predictions_array):2.1f}% ({classes[true_label]})", color=color)

def plot_value_array(i, classes, predictions_array, true_label):
    predictions_array, true_label = predictions_array.cpu(), true_label[i]
    plt.grid(False)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(len(classes)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    plt.rc('xtick', labelsize=6) # fontsize of the tick labels
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def plot_predictions_true(classes, test_probs, test_label, test_images, nrows, ncols):
    n_im = nrows * ncols
    fig = plt.figure(figsize=(2*2*ncols, 2*ncols))
    for i in range(n_im):
        plt.subplot(nrows, 2*ncols, 2*i+1)
        plot_image(i, classes, test_probs[i], test_label, test_images)
        plt.subplot(nrows, 2*ncols, 2*i+2)
        plot_value_array(i, classes, test_probs[i], test_label)
    plt.tight_layout()
    return fig

def plot_image_two(i, classes, predictions_array, img):
    predictions_array, img = predictions_array.cpu(), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    matplotlib_imshow(img, one_channel=True)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel(f"{classes[predicted_label]} {100 * max(predictions_array):2.1f}%")

def plot_value_array_two(i, classes, predictions_array):
    predictions_array = predictions_array.cpu()
    plt.grid(False)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(len(classes)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    plt.rc('xtick', labelsize=6) # fontsize of the tick labels
    predicted_label = np.argmax(predictions_array)

def plot_model_predictions(classes, pred_probs, pred_images, nrows, ncols):
    n_im = nrows * ncols
    fig = plt.figure(figsize=(2*2*ncols, 2*ncols))
    for i in range(n_im):
        plt.subplot(nrows, 2*ncols, 2*i+1)
        plot_image_two(i, classes, pred_probs[i], pred_images)
        plt.subplot(nrows, 2*ncols, 2*i+2)
        plot_value_array_two(i, classes, pred_probs[i])
    plt.tight_layout()
    return fig

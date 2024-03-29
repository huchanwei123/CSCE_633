import numpy as np
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def compute_Pearson(attr1, attr2):
    # Use scipy to calculate Pearson's coefficient
    # function return:
    #   r: Pearson’s correlation coefficient
    #   p-value: Two-tailed p-value
    # I return for r only
    return stats.pearsonr(attr1, attr2)[0]

def plot_2D(attr1, attr1_label, attr2, attr2_label):
    # plot the 2D scatter figure
    colormap = np.array(['r', 'g', 'b'])
    plt.xlabel(attr1_label)
    plt.ylabel(attr2_label)
    plt.scatter(attr1, attr2)
    plt.show()

def plot_line(data, label, xy_name):
    """
    Plot muliple lines using matplotlib
    Input:
        data:    list of data
        label:   list of label for each data
        xy_name: name for x and y axis
    Return:
        Null
    """
    assert len(data) == len(label)
    x_len = len(data[0])
    marker_list = ['-o', '-x', '-*', '-<']

    for i in range(len(data)):
        plt.plot(range(x_len), data[i], marker_list[i], label=label[i])
        #plt.title('Fold {}'.format(i+1))
    plt.xlabel(xy_name[0]) 
    plt.ylabel(xy_name[1])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

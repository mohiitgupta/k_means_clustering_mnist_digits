import numpy as np
from numpy import genfromtxt
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (7,7)


def main():
    raw_digits = genfromtxt('digits-raw.csv', delimiter=',')
    for i in range(10):
        class_i_digits = raw_digits[raw_digits[:,1]==i]
        digit = np.random.choice(len(class_i_digits))
        plt.imsave(str(i)+'.png',class_i_digits[digit][2:].reshape((28,28)),format='png')

    digits_embedding = genfromtxt('digits-embedding.csv', delimiter=',')

    visualize_egs = np.random.randint(0,len(digits_embedding), size=1000)

    class_labels = [0,1,2,3,4,5,6,7,8,9]
    for i in range(10):
        x_axis = []
        y_axis = []
        for eg in visualize_egs:
            digit_array = digits_embedding[eg]
            if digit_array[1] == i:
                x_axis.append(digit_array[2])
                y_axis.append(digit_array[3])
        plt.scatter(x_axis, y_axis, label=i)
    plt.legend(class_labels, loc='best', fontsize=8, bbox_to_anchor=(1, 1))
    # plt.show()
    plt.savefig("exploration2")

if __name__ == '__main__':
    main()
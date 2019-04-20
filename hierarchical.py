import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (7,7)
import random
import sys
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from kmeans import get_wc_ssd, get_SC, get_NMI


def plot_dendogram(Z, max_d, annotate_above, fig_name):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('digit id')
    plt.ylabel('distance')
    R=dendrogram(
        Z,
    #     truncate_mode='lastp',
    #     p=12,
        leaf_rotation=90.,
        leaf_font_size=8., 
        show_contracted=True,
    )
    for i, d, c in zip(R['icoord'], R['dcoord'], R['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
    if max_d:
        plt.axhline(y=max_d, c='k')
    plt.savefig(fig_name,dpi=300)
    plt.close()
    return R

def plot_graph(x_axis, y_axis, x_axis_label, y_axis_label, legend, fig_name):
    plt.plot(x_axis, y_axis)

    
    x_axis_label = x_axis_label
    y_axis_label = y_axis_label
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(legend, loc='best')

    plt.title(y_axis_label + ' v/s ' + x_axis_label)
#     plt.show()
    plt.savefig(fig_name,dpi=300)
    plt.close()
    # plt.show()

def main():
    digits_embedding = genfromtxt('digits-embedding.csv', delimiter=',')

    np.random.seed(0)
    data = []
    for i in range(10):
        class_i_digits = digits_embedding[digits_embedding[:,1]==i]
        digits = np.random.randint(0, len(class_i_digits), size=10)
        for digit in digits:
            data.append(class_i_digits[digit])
    data = np.array(data)  

    # plt.scatter(data[:,2],data[:,3], c=data[:,1])
    # plt.show() 

    '''
    plot dendograms
    '''
    methods = ['single', 'complete', 'average']
    features = data[:,2:4]
    k_list = [2,4,8,16,32]
    # print (features[:20])
    for method in methods: 
        Z = linkage(features, method=method)

        max_d=500
        plot_dendogram(Z, max_d, 10, 'dendogram_'+method)
        wc_ssd_values = []
        sc_values = []
        
        for k in k_list:

            cluster_indices = cut_tree(Z,k)
            features_labels = np.column_stack((features, cluster_indices))
            centroids = {}
            for cluster_id in range(k):
                cluster_members = features[features_labels[:,2] == cluster_id]
                centroid = np.average(cluster_members, axis=0)
                centroids[cluster_id]=centroid
            wc_ssd_values.append(get_wc_ssd(centroids, features, features_labels[:,2]))
            sc_values.append(get_SC(features, features_labels[:,2]))
        # print ("Method", method, "WC-SSD", wc_ssd_values)
        # print ("Method", method, "SC", sc_values)
        plot_graph(k_list, wc_ssd_values, 'k (number of clusters)','WC-SSD',['Sub Sample 100 images, method '+method], 'hierarchical_learning_curve_wc_ssd_'+str(method))
        plot_graph(k_list, sc_values, 'k (number of clusters)','SC',['Sub Sample 100 images, method '+method], 'hierarchical_learning_curve_sc_'+str(method))


    print ("We chose K=8 for all 3 methods single, compelete, average")

    for method in methods: 
        Z = linkage(features, method=method)
        k=8
        cluster_indices = cut_tree(Z,k)
        features_labels = np.column_stack((features, cluster_indices))
        nmi = get_NMI(features, features_labels[:,2], data[:,1])
        print ("For method", method, "NMI:", nmi)

if __name__ == '__main__':
    main()
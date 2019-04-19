from kmeans import get_results_kmeans
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def plot_error_graph(x_axis, y_axis, x_axis_label, y_axis_label, legend, fig_name, se):
    plt.errorbar(x_axis, y_axis, yerr=se, fmt='-', capsize=4, capthick=2)
    x_axis_label = x_axis_label
    y_axis_label = y_axis_label
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(legend, loc='best')

    plt.title(y_axis_label + ' v/s ' + x_axis_label)

def plot_graph(x_axis, y_axis, x_axis_label, y_axis_label, legend, fig_name):
    plt.plot(x_axis, y_axis)

    
    x_axis_label = x_axis_label
    y_axis_label = y_axis_label
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(legend, loc='best')
    plt.title(y_axis_label + ' v/s ' + x_axis_label)

def visualize(data, labels, indices_to_visualize, fig_name):
    clusters = list(np.unique(labels))
    for cluster in clusters:
        x_axis = []
        y_axis = []
        label_list = []
        for index in indices_to_visualize:
            if labels[index] == cluster:
                x_axis.append(data[index][0])
                y_axis.append(data[index][1])
        plt.scatter(x_axis, y_axis, label=cluster)
    plt.legend(clusters, loc='best', fontsize=8, bbox_to_anchor=(1, 1))
    plt.savefig(fig_name)
    plt.close()

def main():
    digits_embedding = genfromtxt('digits-embedding.csv', delimiter=',')
    digits_2 = np.array([2,4,6,7])
    digits_3 = [6,7]
    digits_embedding_2 = digits_embedding[np.in1d(digits_embedding[:,1], digits_2)]
    digits_embedding_3 = digits_embedding[np.in1d(digits_embedding[:,1], digits_3)]
    dataset = []
    dataset.append(digits_embedding)
    dataset.append(digits_embedding_2)
    dataset.append(digits_embedding_3)

    k_values = [2,4,8,16,32]
    # k_values = [2]
    sc_values = []
    wc_ssd_values = []

    '''
    Do analysis for 2.2 part 1.
    '''
    print ("Run analysis for 2.2 part 1")
    for i,digits_embedding in enumerate(dataset):
        features = digits_embedding[:,2:4]
        for k in k_values:
            for seed in range(1):
                print ("run for k", k, "seed", seed, "dataset", i+1)
                wc_ssd, sc, nmi, cluster_indices = get_results_kmeans(features, k, seed, digits_embedding[:,1])
                sc_values.append(sc)
                wc_ssd_values.append(wc_ssd)
    print ("WC-SSD list:", wc_ssd_values)
    print ("SC list:", sc_values)

    legend = ['DATASET 1', '[digits 2, 4, 6, 7]', '[digits 6, 7]']
    for i in range(3):
        starting_index = i*len(k_values)
        plot_graph(k_values, wc_ssd_values[starting_index:starting_index+len(k_values)],
                   "K (number of clusters)", "WC SSD", legend, "learning_curves_wc_ssd_k_dataset_"+str(i))
    
    plt.savefig("learning_curves_wc_ssd_k", dpi=300)
    plt.close()
    for i in range(3):
        starting_index = i*len(k_values)
        plot_graph(k_values, sc_values[starting_index:starting_index+len(k_values)],
                   "K (number of clusters)", "SC", legend, "learning_curves_sc_k_dataset_"+str(i))

    plt.savefig("learning_curves_sc_k", dpi=300)
    plt.close()

    '''
    Do analysis for 2.2 part 3
    '''
    print ("Run analysis for 2.2 part 3")
    avg_sc_values = []
    std_dev_sc_values = []
    avg_wc_ssd_values = []
    std_dev_wc_ssd_values = []
    for i,digits_embedding in enumerate(dataset):
        features = digits_embedding[:,2:4]
        for k in k_values:
            sc_list = []
            wc_ssd_list = []
            for seed in range(10):
                print ("run for k", k, "seed", seed, "dataset", i+1)
                wc_ssd, sc, nmi, cluster_indices = get_results_kmeans(features, k, seed, digits_embedding[:,1])
                sc_list.append(sc)
                wc_ssd_list.append(wc_ssd)
            
            avg_sc_values.append(np.average(sc_list))
            avg_wc_ssd_values.append(np.average(wc_ssd_list))
            std_dev_sc_values.append(np.std(sc_list))
            std_dev_wc_ssd_values.append(np.std(wc_ssd_list))
    print ("Avg WC-SSD list:", avg_wc_ssd_values)
    print ("Avg SC list:", avg_sc_values)
    print ("Standard Deviation WC-SSD list:", avg_wc_ssd_values)
    print ("Standard Deviation SC list:", avg_sc_values)

    for i in range(3):
        starting_index = i*len(k_values)
        plot_error_graph(k_values, avg_wc_ssd_values[starting_index:starting_index+len(k_values)],"K (number of clusters)", "Average WC SSD",
                         legend, "learning_curves_random_seeds_wc_ssd_k_dataset_"+str(i), 
                         std_dev_wc_ssd_values[starting_index:starting_index+len(k_values)])
    plt.savefig("learning_curves_random_seeds_wc_ssd_k")
    plt.close()

    for i in range(3):
        starting_index = i*len(k_values)
        plot_error_graph(k_values, avg_sc_values[starting_index:starting_index+len(k_values)],
                   "K (number of clusters)", "Average SC",legend, "learning_curves_random_seeds_sc_k_dataset_"+str(i), 
                         std_dev_sc_values[starting_index:starting_index+len(k_values)])
    plt.savefig("learning_curves_random_seeds_sc_k")
    plt.close()

    '''
    Visualize for 2.2 part 4
    '''
    print ("\nWe choose k=8 for dataset 1, k=4 for dataset 2, k=2 for dataset 3")
    for i, data in enumerate(dataset):
        if i==0:
            k=8
        elif i==1:
            k=4
        elif i==2:
            k=2
        wc_ssd, sc, nmi, cluster_indices = get_results_kmeans(data[:,2:4], k, 0, data[:,1])
        visualize_egs = np.random.randint(0, len(data), size=1000)
        print ("NMI for dataset", str(i+1), "is",nmi)
        visualize(data[:,2:4],cluster_indices, visualize_egs, 'visualize_dataset_'+str(i+1))


if __name__ == '__main__':
    main()
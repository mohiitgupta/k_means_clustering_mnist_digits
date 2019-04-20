import numpy as np
import sys
from numpy import genfromtxt
import time

def get_squared_distance(p1, p2):
    sq_dist = 0
    for i in range(len(p1)):
        sq_dist +=(p1[i]-p2[i])**2
    return sq_dist

def get_cluster_id(centroids, feature):
    min_dist = float("inf")
    min_centroid = -1
    for i,centroid in enumerate(centroids):
        dist = get_squared_distance(feature, centroid)
        if dist < min_dist:
            min_centroid = i
            min_dist = dist
    
    return min_centroid


def run_kmeans(features, k, seed_value):
    np.random.seed(seed_value)
    centroids = []
    centroid_indices = np.random.randint(0,len(features), size=k)
    for index in centroid_indices:
        centroids.append(features[index])

    cluster_indices = np.zeros(len(features), dtype=np.int8)
    '''
    repeat till 50 iterations
    '''
    num_iterations = 50
    for iteration in range(num_iterations):
    #     print (centroids)
        for i,feature in enumerate(features):
            cluster_indices[i]=get_cluster_id(centroids, feature)

        if iteration == num_iterations-1:
            break
        mean_x_centroids = np.zeros(k)
        mean_y_centroids = np.zeros(k)
        count_points_clusters = np.zeros(k)
        for i, feature in enumerate(features):
            mean_x_centroids[cluster_indices[i]] += feature[0]
            mean_y_centroids[cluster_indices[i]] += feature[1]
            count_points_clusters[cluster_indices[i]] += 1
        
        for i in range(k):
            if count_points_clusters[i] != 0:
                mean_x_centroids[i]/=count_points_clusters[i]
                mean_y_centroids[i]/=count_points_clusters[i]
        for i in range(len(centroids)):
            centroids[i]=[mean_x_centroids[i], mean_y_centroids[i]]
    return cluster_indices, centroids


def get_wc_ssd(centroids, features, cluster_indices):
    wc_ssd=0
    for i, feature in enumerate(features):
        centroid = centroids[cluster_indices[i]]
        wc_ssd += get_squared_distance(feature, centroid)
    return wc_ssd


'''
Calculate silhoutte coefficient
'''
def get_SC(features, cluster_indices):
    s_i_list = np.zeros(len(features))

    features_norm = np.linalg.norm(features, axis=1)**2
    distance_squared = features_norm.reshape(-1,1) + features_norm.reshape(1,-1) - 2*np.dot(features, features.T)
    distance_squared[distance_squared<0]=0
    distance_matrix = np.sqrt(distance_squared)
    
    for i, distance_i in enumerate(distance_matrix):
        same_cluster_distances = distance_i[cluster_indices==cluster_indices[i]]
        if (len(same_cluster_distances)==1):
            s_i_list[i]=0
        else:
            A=np.sum(same_cluster_distances)/(len(same_cluster_distances)-1)

            diff_cluster_distances = distance_i[cluster_indices!=cluster_indices[i]]
            B=np.sum(diff_cluster_distances)/(len(diff_cluster_distances))

            s_i_list[i]=(B-A)/max(A,B)

    SC = np.sum(s_i_list)/len(s_i_list)
    return SC

def get_entropy(labels):
    elements, counts = np.unique(labels, return_counts = True)
    counts=counts/len(labels)
    return elements, -np.sum(counts*np.log(counts))

def get_NMI(features, cluster_indices, class_labels):
    class_unique_labels, class_entropy = get_entropy(class_labels)
    cluster_unique_labels, cluster_entropy = get_entropy(cluster_indices)
    '''
    calculate conditional entropy for class labels given cluster
    '''
    conditional_entropy = 0
    for cluster_id in cluster_unique_labels:
        class_filter = class_labels[cluster_indices==cluster_id]
        class_filter_labels, class_filter_entropy = get_entropy(class_filter)
#         print (len(class_filter))
        conditional_entropy += len(class_filter)*class_filter_entropy
    conditional_entropy /= len(class_labels)
    mutual_information = class_entropy - conditional_entropy
    nmi = mutual_information/(class_entropy+cluster_entropy)
    return nmi

def get_results_kmeans(features, k, seed, class_labels):
    cluster_indices, centroids = run_kmeans(features, k, seed)
    wc_ssd = get_wc_ssd(centroids, features, cluster_indices)
    sc = get_SC(features, cluster_indices)
    nmi = get_NMI(features, cluster_indices, class_labels)
    return wc_ssd, sc, nmi, cluster_indices

def main():
    t0 = time.time()
    if len(sys.argv) != 3:
        print ("usage: python [filename] [dataFilename] [k_value]")
    else:
        digits_embedding = genfromtxt(sys.argv[1], delimiter=',')
        k=int(sys.argv[2])
        seed = 0
        features = digits_embedding[:,2:4]
        wc_ssd, sc, nmi, cluster_indices = get_results_kmeans(features, k, seed, digits_embedding[:,1])
        print("WC-SSD:",wc_ssd)
        print("SC:",sc)
        print("NMI:",nmi)

        
    t1 = time.time()
    total = t1-t0
    # print("total time taken for running code", total, "seconds")
    

if __name__ == '__main__':
    main()
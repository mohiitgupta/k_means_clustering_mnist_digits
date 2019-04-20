The code has been written using python 3. This code has implementation for kmeans algorithm and hierarchical/agglomerative clustering using scipy library. The kmeans code is present in `kmeans.py`.

Note, that the definition of SC used is as per the lecture notes of CS573 by Prof. Ming Yin. This definition is different from the definition of SC as given in wikipedia. Using the definition taught in the lecture, the plot for WC-SSD is an elbow curve but the plot for SC is a monotonically increasing curve as can be seen in report `CS573_HW5.pdf`.

A brief Summary is we run kmeans algorithm on 3 versions of the MNIST dataset. The 1st version contains embeddings of images of all 9 digits, the second version contains only 4 digits. Lastly, the 3rd version contains only 2 digits. We observe that if we vary k from `[2,4,8,16,32]` as per homework instructions, we choose k=8 for the first version of the dataset, k=4 for 2nd version, k=2 for 3rd version. We use WC SSD, SC and NMI for evaluations.

For Hierarchical clustering, we use scipy library. We plot dendograms using the `dendogram` module. We use `linkage` module which contains the core logic to perform hierarchical clustering. We use `cut_tree` module for getting cluster ids after cutting the tree after we get `k` clusters which is an argument to the `cut_tree` module.

The code for analysis for kmeans is present in `kmeans_analysis.py`.
The code for plotting dendograms, curves and analysis for hierarchical clustering is in `hierarchical.py`.

The dendogram plots given in the report are not of very high quality images. For analyzing dendograms, kindly take a look at the image files `dendogram_single.png`, `dendogram_complete.png` and `dendogram_average.png`.

The runtime for kmeans.py is around 150 seconds. The usage is given below:
`python kmeans.py digits-embedding.csv 10`.

Happy Coding!
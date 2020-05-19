from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


def plot_clustering(data):
	'''
		Definition:
			This function plot the squared error for the clustered points
		args:
			data to be clusterd
		returns:
			None
		
	'''	
	cost =[] 
	max_clusters = 20
	for i in range(2, max_clusters):
	    print("Analysing ", i, " clusters")
	    KM = MiniBatchKMeans(n_clusters = i,batch_size=20000) 
	    KM.fit(data)  
	    cost.append(KM.inertia_)    
	  

	plt.plot(range(2, max_clusters), cost, color ='g', linewidth ='3') 
	plt.xlabel("Number of Clusters") 
	plt.ylabel("Squared Error (Cost)") 
	plt.show()
	 

def do_clustering(data,number_clusters):
	'''
		Definition:
			This function initizalize KMeans with number_clusters and fit to data
		args:
			data to be clustered, number_clusters
		returns:
			fitted K-Means mdel
		
	'''	
	
	kmeans = KMeans(number_clusters)
	fitted_model_k_means = kmeans.fit(data)
	return fitted_model_k_means


#pca model n componentes
from sklearn.decomposition import PCA
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
import pandas as pd


def pca_model_n_components(df,n_components):
    '''
	Definition:
		Initialize pca with n_components
	args:
		dataframe and number of components
	returns:
		pca initialized and pca fitted and transformed
		
    '''

    pca = PCA(n_components)
    return pca,pca.fit_transform(df)


def pca_model(df):
    '''
	Definition:
		Initialize pca
	args:
		dataframe
	returns:
		pca initialized and pca fitted and transformed
		
    '''	
    pca = PCA()
    return pca,pca.fit_transform(df)
    
def get_min_components_variance(df,retain_variance):
    '''
	Definition:
		get min components to retain variance
	args:
		dataframe and retained_variance ratio
	returns:
		number of min components to retain variance
		
    '''	
    pca,pca_tranformed = pca_model(df)
    cumulative_sum = np.cumsum(pca.explained_variance_ratio_)
    return min(np.where(cumulative_sum>=retain_variance)[0]+1)

def plot_curve_min_components_variance(df,mode="cumulative_variance"):
    '''
	Definition:
		plot curve of variance of pca
	args:
		dataframe and mode to be plotted (cumulative_variance or variance)
	returns:
		None, only plot the curve
		
    '''	

    rcParams['figure.figsize'] = 12, 8

   
    pca,pca_transformed = pca_model(df)           
    fig = plt.figure()
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_sum = np.cumsum(explained_variance)
    
    n_components = len(explained_variance)
    ind = np.arange(n_components)
    
    
    ax = plt.subplot(111)
    if(mode=="cumulative_variance"):
        title = "Explained Cumulative Variance per Principal Component"
        ylabel = "Cumulative Variance (%)"
        ax.plot(ind, cumulative_sum)
        mark_1 = get_min_components_variance(df,0.2)
        mark_2 = get_min_components_variance(df,0.4)
        mark_3 = get_min_components_variance(df,0.6)
        mark_4 = get_min_components_variance(df,0.8)
        mark_5 = get_min_components_variance(df,0.9)
        mark_6 = get_min_components_variance(df,0.95)
        mark_7 = get_min_components_variance(df,0.99)





        plt.hlines(y=0.2, xmin=0, xmax=mark_1, color='green', linestyles='dashed',zorder=1)
        plt.hlines(y=0.4, xmin=0, xmax=mark_2, color='green', linestyles='dashed',zorder=2)
        plt.hlines(y=0.6, xmin=0, xmax=mark_3, color='green', linestyles='dashed',zorder=3)
        plt.hlines(y=0.8, xmin=0, xmax=mark_4, color='green', linestyles='dashed',zorder=4)
        plt.hlines(y=0.9, xmin=0, xmax=mark_5, color='green', linestyles='dashed',zorder=5)
        plt.hlines(y=0.95, xmin=0, xmax=mark_6, color='green', linestyles='dashed',zorder=6)
        plt.hlines(y=0.99, xmin=0, xmax=mark_7, color='green', linestyles='dashed',zorder=6)


        plt.vlines(x=mark_1, ymin=0, ymax=0.2, color='green', linestyles='dashed',zorder=7)
        plt.vlines(x=mark_2, ymin=0, ymax=0.4, color='green', linestyles='dashed',zorder=8)
        plt.vlines(x=mark_3, ymin=0, ymax=0.6, color='green', linestyles='dashed',zorder=9)
        plt.vlines(x=mark_4, ymin=0, ymax=0.8, color='green', linestyles='dashed',zorder=10)
        plt.vlines(x=mark_5, ymin=0, ymax=0.9, color='green', linestyles='dashed',zorder=11)
        plt.vlines(x=mark_6, ymin=0, ymax=0.95, color='green', linestyles='dashed',zorder=12)
        plt.vlines(x=mark_7, ymin=0, ymax=0.99, color='green', linestyles='dashed',zorder=12)
    else:
        title = "Variance per Principal Component"
        ylabel = "Variance (%)"
        ax.plot(ind, explained_variance)
    
        

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel(ylabel)
    plt.title(title)
    
    


def report_features(feature_names,pca,component_number):
    '''
	Definition:
		This function returns the weights of the original features in relation to a component number of pca
	args:
		feature_names, pca model and the component_number
	returns:
		data frame with features names and the correspondent weights
		
    '''	
    
    components = pca.components_
    
    feature_weights = dict(zip(feature_names, components[component_number]))
    sorted_weights = sorted(feature_weights.items(), key = lambda kv: kv[1])
    
    data = []
    
    
    for feature, weight, in sorted_weights:
        data.append([feature,weight])
    
    df =  pd.DataFrame(data,columns=["feature","weight"])    
    df.set_index("feature",inplace=True)
    return df
        
        
        
        
        
        

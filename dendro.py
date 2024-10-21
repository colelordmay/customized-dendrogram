import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import matplotlib.patches as patches
import matplotlib


def cluster_relabel(y: np.ndarray) -> np.ndarray:
    '''Reassign cluster labels such that the biggest cluster is cluster 0, next biggest is cluster 1, etc.
    
    Parameters
    ----------
    y : ndarray
        Array containing class labels as integers from 0 to n-1, where n is the number of clusters
        
    Returns
    -------
    arr[y] : ndarray
        Relabelled array
        
    Example
    -------
    >>> y_labels = np.array([2,2,2,0,1,1,2])
    >>> cluster_relabel(y_labels)
    array([0,0,0,2,1,1,0])
        
    '''
    n_clusters = len(set(y))
    arr = np.zeros(n_clusters + 1, dtype=int)
    arr[np.arange(n_clusters)[np.bincount(y).argsort()]] = np.arange(n_clusters)[::-1]
    return arr[y]

def decompose_linkage(k,Z,n_obs):
    if k<n_obs:
        yield k
    else:
        for i in Z[k-n_obs][:2]:
            yield from decompose_linkage(i,Z,n_obs)

def improved_dendrogram(ax: plt.axis,
                        dat: np.ndarray,
                        n_clust: int,
                        n_leaves: int=20,
                        c_list: list = None,
                        cluster_names: list = None,
                        rectangle_on = True,
                        cmap = matplotlib.colormaps.get_cmap('Spectral')) -> tuple[plt.axis, np.ndarray]:
    
    '''Using shc and ward's method, plot a dendrogram that is more customizable and visually appealing
    
    Parameters
    ----------
    ax : axis
        Axis object to plot on
        
    dat : ndarray
        2-dimensional numpy array to on which to cluster.
        Should be oriented such that dat.cluster = (n_observations,n_variables)
        
    n_clust : int
        Desired number of clusters
        
    n_leaves : int
        Number of leaves at bottom of dendrogram
        
    c_list : list
        Colors to assign to each cluster. len(c_list) > n is required
        
    cluster_names : list
        Names to label under each cluster

    rectangle_on : bool
        If true, plots a low alpha rectangle underneath each branch to visually highlight each cluster

    cmap : LinearSegmentedColormap
        The colormap to use if c_list is not provided
        
        
    Returns
    -------
    ax : axis
        Axis object containing dendrogram
    
    clus: ndarray
        Array of cluster labels

    '''
    ### This is the only error catch I've bothered to write. There are many other ways to break the code, but this is the most common.
    if c_list:
        if n_clust > len(c_list):
            raise ValueError('If entering a list of colors, len(c_list) >= n_clust is required.')

    ### Correct inputs to reduce errors
    n_obs = dat.shape[0]

    if c_list:
        c_list = np.array(c_list)

    else:
        c_list = cmap(np.linspace(0.1,0.9,n_clust))
    
    
    ### First, cluster as normal. 
    ### We close out of the dendrogram because we're going to plot it ourselves. We just need dend
    Z = shc.linkage(dat, method = 'ward')
    dend = shc.dendrogram(Z, 
            orientation='top', 
            truncate_mode='lastp',
            count_sort='descending',
            p=n_leaves)
    plt.cla()
    
    ### Relabel the clusters such that the first one is the one with the most elements
    clus = cluster_relabel(shc.fcluster(Z,n_clust,criterion='maxclust')-1)

    ### Cast lists as arrays for more convenience
    y_c = np.array(dend['dcoord'])
    x_c = np.array(dend['icoord'])
    Z = np.array(Z,dtype=int)
    
    ### For each of the lowest leaves determine to which cluster it belongs
    ### Run next() on generator because we only need one as by definition all obs in one leaf are 
    CC = clus[np.array([next(decompose_linkage(i,Z,n_obs)) for i in dend['leaves']])]

    ### Determine what color belongs between what horizontal extent
    ### e.g. if c_lims[2] = [12.5,22.5], then all observations between these numbers should be c_list[2]
    c_lims = []
    for i in range(n_clust):
        r = (5+10*np.arange(n_leaves))[CC==i]
        c_lims.append(np.array([r[0]-5,r[-1]+5]))
    c_lims = np.array(c_lims)
    
    ### Automatically determine a cutoff given number of clusters
    cutoff = np.mean(sorted(y_c[:,1])[::-1][n_clust-2:n_clust])

    
    
    ### Add elements to ax object
    ax.axhline(cutoff,c="k",ls="--")
    for X,Y in zip(x_c,y_c):

        color_l = c_list[:n_clust][(X[0]>=c_lims[:,0]) & (X[0]<c_lims[:,1])][0]
        color_r = c_list[:n_clust][(X[-1]>=c_lims[:,0]) & (X[-1]<c_lims[:,1])][0]
        
        if all(Y<cutoff): # Color when all lines below cutoff
            ax.plot(X,Y,c=color_l)

        elif all(Y>cutoff): # Color when all lines above cutoff
            ax.plot(X,Y,c='k')

        else:
            ax.plot(X[1:3],Y[1:3],c="k") # Horizontal line above cutoff
            if (Y[0]<cutoff) & (Y[1]>cutoff): # Left-side line breaks over cutoff
                ax.plot(X[:2],[Y[0],cutoff],c=color_l)
                ax.plot(X[:2],[cutoff,Y[1]],c="k")
            else:
                ax.plot(X[:2],Y[:2],c="k")

            if (Y[2]>cutoff) & (Y[3]<cutoff): # Right-side line breaks over cutoff
                ax.plot(X[2:],[Y[3],cutoff],c=color_r)
                ax.plot(X[2:],[cutoff,Y[2]],c="k")
            else:
                ax.plot(X[2:],Y[2:],c="k")

    if rectangle_on:
        for i in range(n_clust):
            ax.add_patch(patches.Rectangle((c_lims[i][0]+1, 0), c_lims[i][1]-c_lims[i][0]-2, cutoff*0.9, facecolor=c_list[i],alpha=0.1,zorder=-2))
    ### Formatting
    ax.set_xticks([])
    ax.set_ylabel("Intercluster distance")
    for i in range(n_clust):
        if cluster_names:
            ax.text(c_lims.mean(axis=1)[i],-0.2,cluster_names[i],ha='center',va='top')
        else:
            ax.text(c_lims.mean(axis=1)[i],-0.2,f'cluster {i+1}',ha='center',va='top')
    ax.set_xlim(0,n_leaves*10)
    ax.set_ylim(ymin=0)
    
    return ax, clus
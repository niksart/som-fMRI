from minisom import MiniSom
import matplotlib.pyplot as plt
from numpy import array, where

# MNIST DATASET
from sklearn import datasets
from sklearn.preprocessing import scale
def import_MNIST():
    # load the digits dataset from scikit-learn
    digits = datasets.load_digits(n_class=10)
    data = digits.data  # matrix where each row is a vector that represent a digit.
    mnist_data = scale(data)
    mnist_labels = digits.target  # num[i] is the digit represented by data[i]
    return mnist_data, mnist_labels

def get_alpha(x, eps, min_density, max_density):
    m = (1-eps)/(max_density-min_density)
    q = (- (min_density * (1-eps)) / (max_density - min_density)) + eps
    return m*x+q

class SomWrapper:
    
    def __init__(self, som_shape, data_dimension, sigma=1.0, 
                 learning_rate=0.5, neighborhood_function="gaussian", 
                 distance_metric="euclidean", random_seed=None, scaling_factor_plot=6):
        self.som_shape = som_shape
        self.sf = scaling_factor_plot
        self.som = MiniSom(som_shape[0], som_shape[1],
                           data_dimension,
                           sigma=sigma,
                           learning_rate=learning_rate,
                           neighborhood_function=neighborhood_function,
                           distance_metric=distance_metric)
    
    def train(self, data, iterations=5000):
        self.data = data
        
        self.som.pca_weights_init(data)
        print("Training...")
        self.som.train_random(data, iterations)  # random training
        print("\n...ready!")
        
        # map is a dictionary => list of datapoints to which a prototype is related
        self.map = {}
        for i in range(self.som_shape[0]):
            for j in range(self.som_shape[1]):
                self.map[(i,j)] = []
        img_index = 0
        for x in data:  # scatterplot
            w = self.som.winner(x)
            self.map[w] += [img_index]
            img_index += 1

    def plot_labels(self, labels, path=None):
        plt.figure(figsize=((self.som_shape[1]/self.sf), (self.som_shape[0]/self.sf)))
        ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        for x, t in zip(self.data, labels):  # scatterplot
            w = self.som.winner(x)
            plt.text(w[1] + 0.5,  w[0] + 0.5,  str(t),
                     color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
        
        plt.axis([0, self.som.get_weights().shape[1], 0,  self.som.get_weights().shape[0]])
        if path != None:
            plt.savefig(path)
        plt.show()
    
    def plot(self, epsilon, path=None):        
        scatter_data_x = []
        scatter_data_y = []
        densities = []
        for key in self.map.keys():
            densities.append(len(self.map[key]))
            for _ in range(len(self.map[key])):
                scatter_data_x.append(key[0]+1)
                scatter_data_y.append(key[1]+1)
        
        densities = array(densities)
        min_density = densities[where(densities > 0)].min()
        max_density = densities.max()
        
        colors = []
        for key in self.map.keys():
            for _ in range(len(self.map[key])):
                alpha = get_alpha(len(self.map[key]), epsilon, min_density, max_density)
                colors.append([0,0,0,alpha/len(self.map[key])])
        
        plt.figure(figsize=((self.som_shape[1]/self.sf), (self.som_shape[0]/self.sf)))
        ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.axis([0, self.som.get_weights().shape[1]+1, 0,  self.som.get_weights().shape[0]+1])
        plt.scatter(scatter_data_y, scatter_data_x, c=colors)
        plt.show()

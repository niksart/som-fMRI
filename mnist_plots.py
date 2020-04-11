from som_wrapper import SomWrapper, import_MNIST
import os

if not os.path.isdir("mnist_plots"):
    os.makedirs("mnist_plots")

mnist_data, mnist_labels = import_MNIST()

print("Euclidean distance")
sw = SomWrapper((60,60),64,distance_metric='euclidean')
sw.train(mnist_data)
sw.plot_labels(mnist_labels, path="mnist_plots/plot_labels_eucl.png")
sw.plot_density(path="mnist_plots/plot_density_eucl.png")

print("Pearson's distance")
sw = SomWrapper((60,60),64,distance_metric='correlation')
sw.train(mnist_data)
sw.plot_labels(mnist_labels, path="mnist_plots/plot_labels_pearson.png")
sw.plot_density(path="mnist_plots/plot_density_pearson.png")

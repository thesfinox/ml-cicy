# Machine Learning for Complete Intersection Calabi-Yau 3-folds

We consider a machine learning approach to predict the **Hodge numbers** of **Complete Intersection Calabi-Yau 3-folds** in the framework of String Theory using a first [dataset](http://dx.doi.org/10.1088/0264-9381/6/2/006) with the configuration matrices and a dataset containing their [favourable](http://arxiv.org/abs/1708.07907) embedding.

The analysis is divided into different [_Jupyter_](https://jupyter.org/) notebooks (in this example list, hyperlinks return the version for the _original_ dataset):

- the [_preanalysis_](cicy3o_preanalysis.ipynb) contains a detailed visual analysis of the dataset with outliers detection, clustering and PCA performance comparison, feature engineering and features selection,
- the [_classical analysis_](cicy3o_ml.ipynb) deals with the more "classical" approach of machine learning using linear regression models, support vector machines and decision trees,
- the [_ConvNet analysis_](cicy3o_nn.ipynb) uses **convolutional neural networks** to build the appropriate architecture to predict the Hodge numbers starting from the configuration matrix of the manifold,
- the [_transfer learning analysis_](cicy3o_nn_transfer-learning.ipynb) applies a more refined architecture to the feature engineered set using **transfer learning** from the previous convolutional models,
- the [_stacking analysis_](cicy3o_stack.ipynb) is an attempt at stacking ensemble learning to improve the results of the previous analysis.

Each [IPython](https://ipython.org/) notebook is entirely independent and can be run separately. The only real requirement is to first run (at least once) the _preanalysis_ notebook to generate the "analysis-ready" dataset.

Notice that there are several versions of _apparently_ the same notebooks: those named _cicy3o_... refer to the original dataset, while _cicy3f_... to the dataset with **favourable** embeddings. We also present data for the original dataset using only [half of the training set](cicy3o_ml_half_training.ipynb) and the prediction of the [logarithm of the second Hodge number](cicy3o_nn_logh21.ipynb) using neural networks.

## Installation Prerequisites

In order to run the analysis you will need a _Jupyter_ installation (we used an [_Anaconda_](https://anaconda.org/) environment) using **Python 3.6** at least. Moreover you will be required to install the following packages:

- [_Matplotlib_](https://matplotlib.org/)
- [_Numpy_](https://numpy.org)
- [_Pandas_](https://pandas.pydata.org/)
- [_Scikit-learn_ (>= 2.2.4)](https://scikit-learn.org/)
- [_Scikit-optimize_](https://scikit-optimize.github.io/stable/)
- [_Scipy_](https://www.scipy.org/)
- [_Tensorflow (>= 2.0.0)_](https://www.tensorflow.org/)

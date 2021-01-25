# Bio CorEx: recover latent factors with Correlation Explanation (CorEx)

The principle of Total *Cor*-relation *Ex*-planation has recently been introduced as a way to reconstruct latent factors
that are informative about relationships in data. This project consists of python code to build these representations.
While the methods are domain-agnostic, the version of CorEx presented here was designed to handle challenges inherent
in several biomedical problems: missing data, continuous variables, and severely under-sampled data. 

A preliminary version of the technique is described in this paper.      
[*Discovering Structure in High-Dimensional Data Through Correlation Explanation*](http://arxiv.org/abs/1406.1222), 
NIPS 2014.     
This version uses theoretical developments described here:      
[*Maximally Informative Hierarchical Representions of High-Dimensional Data*](http://arxiv.org/abs/1410.7404), 
AISTATS 2015.       
Finally, the Bayesian approach implemented here resulted form work with Shirley Pepke and is described here:     
[*Comprehensive discovery of subsample gene expression components by information explanation: therapeutic implications in cancer*](http://biorxiv.org/content/early/2016/09/19/043257), 
in BMC Medical Genomics (accepted).     
You can also see applications of this code to neuroscience data
[*here*](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=2600367), 
[*here*](https://www.researchgate.net/profile/Madelaine_Daianu2/publication/299377637_Relative_Value_of_Diverse_Brain_MRI_and_Blood-Based_Biomarkers_for_Predicting_Cognitive_Decline_in_the_Elderly/links/56f2b0bd08aea5a8982ff958.pdf), 
and [*here.*](https://www.researchgate.net/profile/Madelaine_Daianu2/publication/305726530_Information-Theoretic_Clustering_of_Neuroimaging_Metrics_Related_to_Cognitive_Decline_in_the_Elderly/links/57a8ab9d08aed76703f87777.pdf)

For sparse binary data, try [*CorEx topic*](http://github.com/gregversteeg/corex_topic/). 
There is also a [linear CorEx](https://github.com/gregversteeg/LinearCorex) that is quite fast, and in the paper, [Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.03353), we showed that CorEx exhibits unique advantages in under-sampled, high-dimensional data. 

### Dependencies

CorEx only requires numpy and scipy. If you use OS X, I recommend installing the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/).

The visualization capabilities in vis_corex.py require other packages: 
* matplotlib - Already in scipy superpack.
* seaborn
* pandas
* [networkx](http://networkx.github.io)  - A network manipulation library. 
* [graphviz](http://www.graphviz.org) (Optional, for compiling produced .dot files into pretty graphs. The command line 
tools are called from vis_corex. Graphviz should be compiled with the triangulation library *gts* for best visual results).

### Install

To install, download using [this link](https://github.com/gregversteeg/bio_corex/archive/master.zip) 
or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/bio_corex.git
```
Use *git pull* to get updates. The code is under development. 
Please contact me about issues.  

### Issue to be resolved

There's a line in the corex.py that says:
```
sig_ml = sig_ml.clip(0.25)
```
This was added after the 2015 AISTATS paper and it helped to avoid numerical errors on gene expression datasets. 
However, thanks to a diligent user, we discovered that the finance results in the AISTATS paper don't reproduce if you have this line.
Removing it returns the original behavior (but also leads to errors on some datasets). 
I plan to make this line into an optional hyper-parameter, and then add a discussion to Troubleshooting to invoke this if there are numerical errors. 
Also see the discussion in the pull requests from a contributor who found in his dataset that making this quantity bigger was sometimes helpful. 
In the meantime, I want people to be aware of it, and consider commenting out the line if you are getting poor results or making it bigger if you are getting numerical errors.

## Example usage with command line interface

These examples use data included in the tests folder. `python vis_corex.py -h` gives an overview of the options. 

`python vis_corex.py data/test_data.csv`

In this simple example there are five variables, v1...v5 where v1, v2, v3 are in one cluster and v4, v5 are in another. 
Looking in "corex_output/graphs" you should see pdf files with the graphs (with graphviz). 

`python vis_corex.py data/test_big5.csv --layers=5,1 --missing=-1 -v --no_row_names -o big5`

This reads the CSV file containing some Big-5 personality survey data. It uses 5 hidden units (and associated clusters)
 at the first layer, and 1 at the second layer. 
 Option -v gives verbose outputs. By default, it is assumed that the first column and row are labels, 
this expectation can be changed with options. There are a few missing values specified with -1. Note that for discrete
data, the discrete values each variable takes have to be like 0,1,...
Finally, all the output files are placed in the directory "big5". 

Looking in the directory "big5/graphs", you should see a pdf that shows the questions clustered into five groups. See
the [full raw data](http://personality-testing.info/_rawdata/BIG5.zip) for information on questions. "Text_files" 
summarizes the clusters and gives the latent factor (or personality trait in this case) associated with each cluster, 
for each sample (survey taker). 

Here are the options for the gene expression data cited above. 
```
python vis_corex.py data/matrix.tcga_ov.geneset1.log2.varnorm.RPKM.txt --delimiter=' ' --layers=200,30,8,1 --dim_hidden=3 --max_iter=100 --missing=-1e6 -c -b -v -o output_folder --ram=8 --cpu=4
```
The delimiter is set because the data is separated by spaces, not the default commas. 
The layers match our specification for the paper. dim_hidden says that each latent factor can take three states
instead of the default of two. 
The default expectation is discrete data. The -c option is used for continuous data. The *v* is for verbose output.
This dataset has a small number of samples so we turned on bayesian smoothing with -b, although this slows down computation.
The ram is approximate ram in GB of your machine and setting cpu=n lets you use however many cpus/cores are on your machine. 
This took me over a day to run. You can try something faster by using layers=20,5,1 and reducing max_iter. 

Also look in the "relationships" folder to look at pairwise relationships between variables in the same group. You should 
see that the variables are strongly dependent. The plot marker corresponds to the latent factor Yj for that group, and
the point color corresponds to p(yj|x) for that point.


## Python API usage

### Example

The API design is based on the scikit-learn package. You define a model (`model=Corex(with options here)`) then use
 the `model.fit(data)` method to fit it on data, then you can transform new data with `model.transform(new_data)`. 
 The model has many other methods to access mutual information, measures of TC, and more. 
```python
import corex as ce

X = np.array([[0,0,0,0,0], # A matrix with rows as samples and columns as variables.
              [0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)

layer1 = ce.Corex(n_hidden=2, dim_hidden=2, marginal_description='discrete', smooth_marginals=False)  
# Define the number of hidden factors to use (n_hidden=2). 
# And each latent factor is binary (dim_hidden=2)
# marginal_description can be 'discrete' or 'gaussian' if your data is continuous
# smooth_marginals = True turns on Bayesian smoothing
layer1.fit(X)  # Fit on data. 

layer1.clusters  # Each variable/column is associated with one Y_j
# array([0, 0, 0, 1, 1])
layer1.labels[:, 0]  # Labels for each sample for Y_0
# array([0, 0, 1, 1])
layer1.labels[:, 1]  # Labels for each sample for Y_1
# array([0, 1, 0, 1])
layer1.tcs  # TC(X;Y_j) (all info measures reported in nats). 
# array([ 1.385,  0.692])
# TC(X_Gj) >=TC(X_Gj ; Y_j)
# For this example, TC(X1,X2,X3)=1.386, TC(X4,X5) = 0.693
```

Suppose you want to transform some test data (that wasn't used in training). Assume that `X_test` is a matrix of such data. 
*Make sure the columns of `X_test` exactly match the columns of X.*  If the test data doesn't include all the same columns, you can specify missing values. Check `layer1.missing_values` or set missing_values=(some number) at training time so that you know how to set missing values in training data. For instance, the default is missing_values=-1, then you an put a -1 in your test data for any column that is missing. 
Continuing the example above, you would generate labels on test data as follows. 
```python
X_test = np.array([[1,1,1,0,1]])  # 1 sample/row of data with the same 5 columns as example above
y = layer1.transform(X_test, details=False)
# array([[1, 0]]), I.e., Y_0 = 1 and Y_1 = 0
p, log_z = layer1.transform(X_test, details=True)
# p is p(yj | x), the probability of each factor taking a certain value.
# The shape is n_hidden, number of samples, dim_hidden
#array([[[ 0. ,  1. ]],
#      [[ 0.5,  0.5]]])
# So Y_0 takes values 1 with probability 1 (it matches the training example)
# But Y_1 takes values 0 or 1 with probability 0.5 (it is a mix of the two training examples)
test_tcs = np.mean(log_z[:,:,0], axis=1)
# array([ 1.385, -6.216])
# Compare this value to layer1.tcsl = array([ 1.385,  0.692])
# This tells us that the "test TC" for factor Y_0 is similar to the training data, 
# but the "test TC" for factor Y_1 is completely different. 
```
I interpret negative values in the test TC as meaning *correlations that appeared in the training data do not appear in the test data*. But we haven't studied this phenomena in depth. If you get a bunch of negative values for test TC, though, it signals a significant difference between your training and testing data. 


### Data format

You can specify the type of the variables by passing the option `marginal_description='discrete'` for discrete variables or
`marginal_description='gaussian'` for continuous variables. 
For the discrete version of CorEx, you must input a matrix of integers whose rows represent samples and whose columns
represent different variables. The values must be integers `{0,1,...,k-1}` where k represents the maximum number of 
values that each variable, x_i can take. By default, entries equal to -1 are treated as missing. This can be 
altered by passing a *missing_values* argument when initializing CorEx. 
"smooth_marginals" tells whether to use Bayesian shrinkage estimators for marginal distributions to reduce noise.
It is turned on by default but is off in the example above (since it only has 4 samples, the smoothing will mess it up).

### CorEx outputs

As shown in the example, `clusters` gives the variable clusters for each hidden factor `Y_j` and 
`labels` gives the labels for each sample for each `Y_j`. 
Probabilistic labels can be accessed with `p_y_given_x`. 

The total correlation explained by each hidden factor, `TC(X;Y_j)`, is accessed with `tcs`. Outputs are sorted
so that Y_0 is always the component that explains the highest TC. 
Like point-wise mutual information, you can define point-wise total correlation measure for an individual sample, `x^l`     
`TC(X = x^l;Y_j) == log Z_j(x)`   
This quantity is accessed with `log_z`. This represents the correlations explained by `Y_j` for an individual sample.
A low (or even negative!) number can be obtained. This can be interpreted as a measure of how surprising an individual
observation is. This can be useful for anomaly detection. 

See the main section of vis_corex.py for more ideas of how to do visualization.

## Details

### Computational complexity

This version has time and memory requirements like O(num. samples * num. variables * num. hidden units). By implementing
 mini-batch updates, we could eliminate the dependence on the number of samples. Sorry I haven't gotten to this yet. I
 have been able to run examples with thousands of variables, thousands of samples, and 100 latent factors on my laptop.
 It might also be important to check that your numpy implementation is linked to a good linear algebra library like 
 lapack or BLAS. 

### Hierarchical CorEx
The simplest extension is to stack CorEx representations on top of each other. 
```
layer1 = ce.Corex(n_hidden=100)
layer2 = ce.Corex(n_hidden=10)
layer3 = ce.Corex(n_hidden=1)
Y1 = layer1.fit_transform(X)
Y2 = layer2.fit_transform(Y1.labels)
Y3 = layer2.fit_transform(Y2.labels)
```
The sum of total correlations explained by each layer provides a successively tighter lower bound on TC(X) (see AISTATS paper). 
 To assess how large your representations should be, look at quantities
like layer.tcs. Do all the Y_j's explain some correlation (i.e., all the TCs are significantly larger than 0)? If not
you should probably use a smaller representation.

### Missing values
You can set missing values (by specifying missing_values=-1, when calling, e.g.). CorEx seems robust to missing data.
This hasn't been extensively tested yet though, and we don't really understand the 
effect of data missing not at random. 

### Getting better results
You can use  the option smooth_marginals to turn on the use of Bayesian smoothing methods (off by default) for 
estimating the marginal distributions. This is slower, but reduces spurious correlations, especially if the number
of samples is small (less than 200) or the number of variables or dim_hidden are big. 

Also note that CorEx can find different local optima after different random restarts. You can run it k times and take
the best solution with the "repeat" option. 

Warning: in recent experiments on gene expression that contained lots of zero counts, we got bad results. (The paper had removed columns that included zero counts.)  I'm not sure what the underlying cause is (bad data versus some issue that CorEx has with zero-inflated data), but I strongly recommend removing columns/genes with lots of zeros. 

### Troubleshooting visualization

For Mac users: 

To get the visualization of the hierarchy looking nice sometimes takes a little effort. To get graphs to compile correctly do the following. 
Using "brew" to install, you need to do "brew install gts" followed by "brew install --with-gts graphviz". 
The (hacky) way that the visualizations are produced is the following. The code, vis_corex.py, produces a text file called "graphs/graph.dot". This just encodes the edges between nodes in dot format. Then, the code calls a command line utility called sfdp that is part of graphviz, 
```
sfdp tree.dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.02 -Gratio=0.7 -Gsplines=True -o nice.pdf
```
These dot files can also be opened with OmniGraffle if you would like to be able to manipulate them by hand. 
If you want, you can try to recompile graphs yourself with different options to make them look nicer. Or you can edit the dot files to get effects like colored nodes, etc. 

Also, note that you can color nodes in the graphs by putting prepending a color to column label names in the CSV file.
For instance, blue_column_1_label_name will show column_1_label_name in blue in the graphs folder. Any matplotlib colors are allowed. 
See the BIG5 data file and graphs produced by the command line utility. 

For Ubuntu users:

Credits: https://gitlab.com/graphviz/graphviz/issues/1237

1. Remove any existing installation with `conda uninstall graphviz`. (If you did not install with Conda, you might need to do `sudo apt purge graphviz` and/or `pip uninstall graphviz`).
    
2. run `sudo apt install libgts-dev`

3. run `sudo pkg-config --libs gts`
    
4. run `sudo pkg-config --cflags gts`

5. Download `graphviz-2.40.1.tar.gz` from [here](https://graphviz.gitlab.io/pub/graphviz/stable/SOURCES/graphviz.tar.gz)

6. Navigate to directory containing download, and extract with `tar -xvf graphviz-2.40.1.tar.gz` (or newer whatever the download is named.)

7. `cd` into extracted folder (ie `cd graphviz-2.40.1`) and run `sudo ./configure --with-gts`

8. Run `sudo make` in the folder

9. Run `sudo make install` in the folder

10. Reinstall library using `pip install graphviz`
    

### Other files that are produced
*text_files/groups.txt*
Lists the variables in each group.

*text_files/labels.txt*
Gives a column for each latent factor (in layer 1) and a row for each patient/sample. The entry is the value of the latent factor (0,…dim_hidden-1)

*text_files/cont_labels.txt*
Gives a continuous number to sort each patient with respect to each latent factor. 

*relationships*
For each latent factor, it shows pairwise plots between the top genes in each group. Each point corresponds to a sample/patient and the color corresponds to the learned latent factor. 

### All options
When you run vis_corex.py with the -h option, you get all the command line options. 
```
python vis_corex.py -h
Usage: vis_corex.py [options] data_file.csv 
It is assumed that the first row and first column of the data CSV file are labels.
Use options to indicate otherwise.

Options:
  -h, --help            show this help message and exit

  Input Data Format Options:
    -c, --continuous    Input variables are continuous (default assumption is
                        that they are discrete).
    -t, --no_column_names
                        We assume the top row is variable names for each
                        column. This flag says that data starts on the first
                        row and gives a default numbering scheme to the
                        variables (1,2,3...).
    -f, --no_row_names  We assume the first column is a label or index for
                        each sample. This flag says that data starts on the
                        first column.
    -m MISSING, --missing=MISSING
                        Treat this value as missing data. Default is -1e6. 
    -d DELIMITER, --delimiter=DELIMITER
                        Separator between entries in the data, default is ','.

  CorEx Options:
    -l LAYERS, --layers=LAYERS
                        Specify number of units at each layer: 5,3,1 has 5
                        units at layer 1, 3 at layer 2, and 1 at layer 3
    -k DIM_HIDDEN, --dim_hidden=DIM_HIDDEN
                        Latent factors take values 0, 1..k. Default k=2
    -b, --bayesian_smoothing
                        Turn on Bayesian smoothing when estimating marginal
                        distributions (p(x_i|y_j)). Slower, but reduces
                        appearance of spurious correlations if the number of
                        samples is < 200 or if dim_hidden is large.
    -r REPEAT, --repeat=REPEAT
                        Run r times and return solution with best TC.

  Output Options:
    -o OUTPUT, --output=OUTPUT
                        A directory to put all output files.
    -v, --verbose       Print rich outputs while running.
    -e MAX_EDGES, --edges=MAX_EDGES
                        Show at most this many edges in graphs.
    -q, --regraph       Don't re-run corex, just re-generate outputs (perhaps
                        with edges option changed).

  Computational Options:
    -a RAM, --ram=RAM   Approximate amount of RAM to use (in GB).
    -p CPU, --cpu=CPU   Number of cpus/cores to use.
    -w MAX_ITER, --max_iter=MAX_ITER
                        Max number of iterations to use.
```

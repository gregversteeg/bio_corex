Changes in vis_corex.py

1. DeprecationWarning: `logsumexp` is deprecated!
    
    * Before: `from scipy.misc import logsumexp`  
    * After: `from scipy.special import logsumexp`

2. UserWarning: The `size` parameter has been renamed to `height`; pleaes update your code.
    
    * Before: `sns.pairplot(subdata, kind="reg", diag_kind="kde", size=5, dropna=True)`
    * After: `sns.pairplot(subdata, kind="reg", diag_kind="kde", height=5, dropna=True)`

Changes in readme.md

3. added `graphviz` installation instructions for Ubuntu users

4. added `requirements.txt` which lets users to install all the dependencies with a single command `conda install --yes --file requirements.txt`

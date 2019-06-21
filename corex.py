"""Maximally Informative Representations using CORrelation EXplanation

Main ideas first described in:
Greg Ver Steeg and Aram Galstyan. "Maximally Informative
Hierarchical Representations of High-Dimensional Data"
AISTATS, 2015. arXiv preprint(arXiv:1410.7404.)

The Bayesian smoothing option is described in:
Pepke and Ver Steeg, Comprehensive discovery of subsample gene expression components
by information explanation: therapeutic implications in cancer. BMC Medical Genomics, 2017.

Code below written by: Greg Ver Steeg (gregv@isi.edu)

License: Apache V2
"""

from __future__ import print_function
import sys
import numpy as np  # Tested with 1.8.0
from os import makedirs
from os import path
from numpy import ma
from scipy.special import logsumexp  # Tested with 1.3.0
from multiprocessing.dummy import Pool


def unwrap_f(arg):
    """Multiprocessing pool.map requires a top-level function."""
    return Corex.calculate_p_xi_given_y(*arg)

def logsumexp2(z):
    """Multiprocessing pool.map requires a top-level function."""
    return logsumexp(z, axis=2)


class Corex(object):
    """
    Correlation Explanation

    A method to learn a hierarchy of successively more abstract
    representations of complex data that are maximally
    informative about the data. This method is unsupervised,
    requires no assumptions about the data-generating model,
    and scales linearly with the number of variables.

    Code follows sklearn naming/style (e.g. fit(X) to train)

    Parameters
    ----------
    n_hidden : int, optional, default=2
        Number of hidden units.

    dim_hidden : int, optional, default=2
        Each hidden unit can take dim_hidden discrete values.

    max_iter : int, optional
        Maximum number of iterations before ending.

    n_repeat : int, optional
        Repeat several times and take solution with highest TC.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
        2 output alpha matrix and MIs as you go.

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    labels : array, [n_hidden, n_samples]
        Label for each hidden unit for each sample.

    clusters : array, [n_visible]
        Cluster label for each input variable.

    p_y_given_x : array, [n_hidden, n_samples, dim_hidden]
        The distribution of latent factors for each sample.

    alpha : array-like, shape (n_components,)
        Adjacency matrix between input variables and hidden units. In range [0,1].

    mis : array, [n_hidden, n_visible]
        Mutual information between each (visible/observed) variable and hidden unit

    tcs : array, [n_hidden]
        TC(X_Gj;Y_j) for each hidden unit

    tc : float
        Convenience variable = Sum_j tcs[j]

    tc_history : array
        Shows value of TC over the course of learning. Hopefully, it is converging.

    References
    ----------

    [1]     Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
            High-Dimensional Data Through Correlation Explanation."
            NIPS, 2014. arXiv preprint(arXiv:1406.1222.)

    [2]     Greg Ver Steeg and Aram Galstyan. "Maximally Informative
            Hierarchical Representations of High-Dimensional Data"
            AISTATS, 2015. arXiv preprint(arXiv:1410.7404.)

    [3]     Pepke and Ver Steeg, Comprehensive discovery of subsample
            gene expression components by information explanation:
            therapeutic implications in cancer. BMC Medical Genomics, 2017.

    """
    def __init__(self, n_hidden=2, dim_hidden=2,            # Size of representations
                 max_iter=100, n_repeat=1, ram=8., max_samples=10000, n_cpu=1,   # Computational limits
                 eps=1e-5, marginal_description='gaussian', smooth_marginals=False,    # Parameters
                 missing_values=-1, seed=None, verbose=False):

        self.dim_hidden = dim_hidden  # Each hidden factor can take dim_hidden discrete values
        self.n_hidden = n_hidden  # Number of hidden factors to use (Y_1,...Y_m) in paper
        self.missing_values = missing_values  # For a sample value that is unknown

        self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence
        self.n_repeat = n_repeat  # Run multiple times and take solution with largest TC
        self.ram = ram  # Approximate amount of memory to use in GB
        self.max_samples = max_samples  # The max number of samples to use for estimating MI and unique info
        self.n_cpu = n_cpu  # number of CPU's to use, None will detect number of CPU's
        self.pool = None  # Spin up and close pool of processes in main loop (depending on self.n_cpu)

        self.eps = eps  # Change in TC to signal convergence
        self.smooth_marginals = smooth_marginals  # Less noisy estimation of marginal distributions

        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose > 0:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print('corex, rep size:', n_hidden, dim_hidden)
        if verbose:
            np.seterr(all='ignore')
            # Can change to 'raise' if you are worried to see where the errors are
            # Locally, I "ignore" underflow errors in logsumexp that appear innocuous (probabilities near 0)
        else:
            np.seterr(all='ignore')
        self.tc_min = 0.01  # Try to "boost" hidden units with less than tc_min. Haven't tested value much.
        self.marginal_description = marginal_description
        if verbose:
            print("Marginal description: ", marginal_description)

    def label(self, p_y_given_x):
        """Maximum likelihood labels for some distribution over y's"""
        return np.argmax(p_y_given_x, axis=2).T

    @property
    def labels(self):
        """Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
        return self.label(self.p_y_given_x)

    @property
    def clusters(self):
        """Return cluster labels for variables"""
        return np.argmax(self.alpha[:, :, 0], axis=0)

    @property
    def tc(self):
        """The total correlation explained by all the Y's.
        """
        return np.sum(self.tcs)

    def fit(self, X):
        """Fit CorEx on the data X. See fit_transform.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit CorEx on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.
        """

        if self.n_cpu == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpu)
        Xm = ma.masked_equal(X, self.missing_values)
        best_tc = -np.inf
        for n_rep in range(self.n_repeat):

            self.initialize_parameters(X)

            for nloop in range(self.max_iter):

                self.log_p_y = self.calculate_p_y(self.p_y_given_x)
                self.theta = self.calculate_theta(Xm, self.p_y_given_x)

                if self.n_hidden > 1:  # Structure learning step
                    self.update_alpha(self.p_y_given_x, self.theta, Xm, self.tcs)

                self.p_y_given_x, self.log_z = self.calculate_latent(self.theta, Xm)

                self.update_tc(self.log_z)  # Calculate TC and record history to check convergence

                self.print_verbose()
                if self.convergence():
                    break

            if self.verbose:
                print('Overall tc:', self.tc)
            if self.tc > best_tc:
                best_tc = self.tc
                best_dict = self.__dict__.copy()  # TODO: what happens if n_cpu > 1 and n_repeat > 1? Does pool get copied? Probably not...just a pointer to the same object... Seems fine.
        self.__dict__ = best_dict
        if self.verbose:
            print('Best tc:', self.tc)

        self.sort_and_output(Xm)
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        return self.labels

    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        Xm = ma.masked_equal(X, self.missing_values)
        p_y_given_x, log_z = self.calculate_latent(self.theta, Xm)
        labels = self.label(p_y_given_x)
        if details == 'surprise':
            # Totally experimental
            log_marg_x = self.calculate_marginals_on_samples(self.theta, Xm, return_ratio=False)
            n_samples = Xm.shape[0]
            surprise = []
            for l in range(n_samples):
                q = - sum([max([log_marg_x[j,l,i,labels[l, j]]
                                for j in range(self.n_hidden)])
                           for i in range(self.n_visible)])
                surprise.append(q)
            return p_y_given_x, log_z, np.array(surprise)
        elif details:
            return p_y_given_x, log_z
        else:
            return labels

    def initialize_parameters(self, X):
        """Set up starting state

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        """
        self.n_samples, self.n_visible = X.shape[:2]
        if self.marginal_description == 'discrete':
            values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
            self.dim_visible = int(max(values_in_data)) + 1
            if not set(range(self.dim_visible)) == values_in_data:
                print("Warning: Data matrix values should be consecutive integers starting with 0,1,...")
            assert max(values_in_data) <= 32, "Due to a limitation in np.choice, discrete valued variables" \
                                              "can take values from 0 to 31 only."
        self.initialize_representation()

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
        pseudo_counts = 0.001 + np.sum(p_y_given_x, axis=1, keepdims=True)
        log_p_y = np.log(pseudo_counts) - np.log(np.sum(pseudo_counts, axis=2, keepdims=True))
        return log_p_y

    def calculate_theta(self, Xm, p_y_given_x):
        """Estimate marginal parameters from data and expected latent labels."""
        theta = []
        for i in range(self.n_visible):
            not_missing = np.logical_not(ma.getmaskarray(Xm)[:, i])
            theta.append(self.estimate_parameters(Xm.data[not_missing, i], p_y_given_x[:, not_missing]))
        return np.array(theta)

    def update_alpha(self, p_y_given_x, theta, Xm, tcs):
        """A rule for non-tree CorEx structure.
        """
        sample = np.random.choice(np.arange(Xm.shape[0]), min(self.max_samples, Xm.shape[0]), replace=False)
        p_y_given_x = p_y_given_x[:, sample, :]
        not_missing = np.logical_not(ma.getmaskarray(Xm[sample]))

        alpha = np.empty((self.n_hidden, self.n_visible))
        n_samples, n_visible = Xm.shape
        memory_size = float(self.max_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_visible / memory_size), 1, n_visible)
        for i in range(0, n_visible, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta[i:i+batch_size], Xm[sample, i:i+batch_size])
            correct_predictions = np.argmax(p_y_given_x, axis=2)[:, :, np.newaxis] == np.argmax(log_marg_x, axis=3)
            for ip in range(i, min(i + batch_size, n_visible)):
                alpha[:, ip] = self.unique_info(correct_predictions[:, not_missing[:, ip], ip - i].T)

        for j in np.where(np.abs(tcs) < self.tc_min)[0]:  # Priming for un-used hidden units
            amax = np.clip(np.max(alpha[j, :]), 0.01, 0.99)
            alpha[j, :] = alpha[j, :]**(np.log(0.99)/np.log(amax)) + 0.001 * np.random.random(self.n_visible)
        self.alpha = alpha[:, :, np.newaxis]  # TODO: This is the "correct" update but it is quite noisy. Add smoothing?

    def unique_info(self, correct):
        """*correct* has n_samples rows and n_hidden columns.
            It indicates whether the ml estimate based on x_i for y_j is correct for sample l
            Returns estimate of fraction of unique info in each predictor j=1...m
        """
        n_samples, n_hidden = correct.shape
        total = np.clip(np.sum(correct, axis=0), 1, n_samples)
        ordered = np.argsort(total)[::-1]

        unexplained = np.ones(n_samples, dtype=bool)
        unique = np.zeros(n_hidden, dtype=int)
        for j in ordered:
            unique[j] = np.dot(unexplained.astype(int), correct[:, j])  # np.sum(correct[unexplained, j])
            unexplained = np.logical_and(unexplained, np.logical_not(correct[:, j]))

        frac_unique = [float(unique[j]) / total[j] for j in range(n_hidden)]
        return np.array(frac_unique)

    def calculate_latent(self, theta, Xm):
        """"Calculate the probability distribution for hidden factors for each sample."""
        n_samples, n_visible = Xm.shape
        log_p_y_given_x_unnorm = np.empty((self.n_hidden, n_samples, self.dim_hidden))
        memory_size = float(n_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_samples / memory_size), 1, n_samples)
        for l in range(0, n_samples, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta, Xm[l:l+batch_size])  # LLRs for each sample, for each var.
            log_p_y_given_x_unnorm[:, l:l+batch_size, :] = self.log_p_y + np.einsum('ikl,ijkl->ijl', self.alpha, log_marg_x, optimize=False)
        return self.normalize_latent(log_p_y_given_x_unnorm)

    def normalize_latent(self, log_p_y_given_x_unnorm):
        """Normalize the latent variable distribution

        For each sample in the training set, we estimate a probability distribution
        over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
        This normalization factor is quite useful as described in upcoming work.

        Parameters
        ----------
        Unnormalized distribution of hidden factors for each training sample.

        Returns
        -------
        p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
            p(y_j|x^l), the probability distribution over all hidden factors,
            for data samples l = 1...n_samples
        log_z : 2D array, shape (n_hidden, n_samples)
            Point-wise estimate of total correlation explained by each Y_j for each sample,
            used to estimate overall total correlation.

        """

        log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
        log_z = log_z.reshape((self.n_hidden, -1, 1))

        return np.exp(log_p_y_given_x_unnorm - log_z), log_z

    def calculate_p_xi_given_y(self, xi, thetai):
        not_missing = np.logical_not(ma.getmaskarray(xi))
        z = np.zeros((self.n_hidden, len(xi), self.dim_hidden))
        z[:, not_missing, :] = self.marginal_p(xi[not_missing], thetai)
        return z  # n_hidden, n_samples, dim_hidden

    def calculate_marginals_on_samples(self, theta, Xm, return_ratio=True):
        """Calculate the value of the marginal distribution for each variable, for each hidden variable and each sample.

        theta: array parametrizing the marginals
        Xm: the data
        returns log p(y_j|x_i)/p(y_j) for each j,sample,i,y_j. [n_hidden, n_samples, n_visible, dim_hidden]
        """
        n_samples, n_visible = Xm.shape
        log_marg_x = np.zeros((self.n_hidden, n_samples, n_visible, self.dim_hidden))  #, dtype=np.float32)
        if n_visible > 1 and self.pool is not None:
            args = zip([self] * len(theta), Xm.T, theta)
            log_marg_x = np.array(self.pool.map(unwrap_f, args)).transpose((1, 2, 0, 3))
        else:
            for i in range(n_visible):
                log_marg_x[:, :, i, :] = self.calculate_p_xi_given_y(Xm[:, i], theta[i])
        if return_ratio:  # Return log p(xi|y)/p(xi) instead of log p(xi|y)
            # Again, I use the same p(y) here for each x_i, but for missing variables, p(y) on obs. sample may be different.
            # log_marg_x -= logsumexp(log_marg_x + self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden)), axis=3)[..., np.newaxis]
            log_marg_x += self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
            if self.pool is not None:
                log_marg_x -= np.array(self.pool.map(logsumexp2, log_marg_x))[..., np.newaxis]
            else:
                log_marg_x -= logsumexp(log_marg_x, axis=3)[..., np.newaxis]
            log_marg_x -= self.log_p_y.reshape((self.n_hidden, 1, 1, self.dim_hidden))
        return log_marg_x

    def initialize_representation(self):
        if self.n_hidden > 1:
            self.alpha = (0.5+0.5*np.random.random((self.n_hidden, self.n_visible, 1)))
        else:
            self.alpha = np.ones((self.n_hidden, self.n_visible, 1), dtype=float)
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)

        p_rand = np.random.dirichlet(np.ones(self.dim_hidden), (self.n_hidden, self.n_samples))
        self.p_y_given_x, self.log_z = self.normalize_latent(np.log(p_rand))

    def update_tc(self, log_z):
        self.tcs = np.mean(log_z, axis=1).reshape(-1)
        self.tc_history.append(np.sum(self.tcs))

    def print_verbose(self):
        if self.verbose:
            print(self.tcs)
        if self.verbose > 1:
            print(self.alpha[:, :, 0])
            print(self.theta)
            if hasattr(self, "mis"):
                print(self.mis)

    def convergence(self):
        if len(self.tc_history) > 10:
            dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
            return np.abs(dist) < self.eps  # Check for convergence.
        else:
            return False

    def __getstate__(self):
        # In principle, if there were variables that are themselves classes... we have to handle it to pickle correctly
        # But I think I programmed around all that.
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def save(self, filename):
        """ Pickle a class instance. E.g., corex.save('saved.dat') """
        import pickle
        if path.dirname(filename) and not path.exists(path.dirname(filename)):
            makedirs(path.dirname(filename))
        pickle.dump(self, open(filename, 'wb'), protocol=-1)

    def load(self, filename):
        """ Unpickle class instance. E.g., corex = ce.Marginal_Corex().load('saved.dat') """
        import pickle
        return pickle.load(open(filename, 'rb'))

    def sort_and_output(self, Xm):
        order = np.argsort(self.tcs)[::-1]  # Order components from strongest TC to weakest
        self.tcs = self.tcs[order]  # TC for each component
        self.alpha = self.alpha[order]  # Connections between X_i and Y_j
        self.p_y_given_x = self.p_y_given_x[order]  # Probabilistic labels for each sample
        self.theta = self.theta[:, :, order, :]  # Parameters defining the representation
        self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
        self.log_z = self.log_z[order]  # -log_z can be interpreted as "surprise" for each sample
        if not hasattr(self, 'mis'):
            # self.update_marginals(Xm, self.p_y_given_x)
            self.mis = self.calculate_mis(self.p_y_given_x, self.theta, Xm)
        else:
            self.mis = self.mis[order]
        bias, sig = self.mi_bootstrap(Xm, n_permutation=20)
        self.mis = (self.mis - bias) * (self.mis > sig)

    def calculate_mis(self, p_y_given_x, theta, Xm):
        mis = np.zeros((self.n_hidden, self.n_visible))
        sample = np.random.choice(np.arange(Xm.shape[0]), min(self.max_samples, Xm.shape[0]), replace=False)
        n_observed = np.sum(np.logical_not(ma.getmaskarray(Xm[sample])), axis=0)

        n_samples, n_visible = Xm.shape
        memory_size = float(n_samples * n_visible * self.n_hidden * self.dim_hidden * 64) / 1000**3  # GB
        batch_size = np.clip(int(self.ram * n_visible / memory_size), 1, n_visible)
        for i in range(0, n_visible, batch_size):
            log_marg_x = self.calculate_marginals_on_samples(theta[i:i+batch_size, ...], Xm[sample, i:i+batch_size])  # n_hidden, n_samples, n_visible, dim_hidden
            mis[:, i:i+batch_size] = np.einsum('ijl,ijkl->ik', p_y_given_x[:, sample, :], log_marg_x, optimize=False) / n_observed[i:i+batch_size][np.newaxis, :]
        return mis  # MI in nats

    def mi_bootstrap(self, Xm, n_permutation=20):
        # est. if p-val < 1/n_permutation = 0.05
        mis = np.zeros((self.n_hidden, self.n_visible, n_permutation))
        for j in range(n_permutation):
            p_y_given_x = self.p_y_given_x[:, np.random.permutation(self.n_samples), :]
            theta = self.calculate_theta(Xm, p_y_given_x)
            mis[:, :, j] = self.calculate_mis(p_y_given_x, theta, Xm)
        return np.mean(mis, axis=2), np.sort(mis, axis=2)[:, :, -2]

    # IMPLEMENTED MARGINAL DISTRIBUTIONS
    # For each distribution, we need:
    # marginal_p_METHOD(self, xi, thetai), to define the marginal probability of x_i given theta_i (for each y_j=k)
    # estimate_parameters_METHOD(self, x, p_y_given_x), a way to estimate theta_i from samples (for each y_j=k)
    # marginal_p should be vectorized. I.e., xi can be a single xi or a list.

    def marginal_p(self, xi, thetai):
        """Estimate marginals, log p(xi|yj) for each possible type. """
        if self.marginal_description == 'gaussian':
            mu, sig = thetai  # mu, sig have size m by k
            xi = xi.reshape((-1, 1, 1))
            return (-(xi - mu)**2 / (2. * sig) - 0.5 * np.log(2 * np.pi * sig)).transpose((1, 0, 2))  # log p(xi|yj)

        elif self.marginal_description == 'discrete':
            # Discrete data: should be non-negative integers starting at 0: 0,...k. k < 32 because of np.choose limits
            logp = [theta[np.newaxis, ...] for theta in thetai]  # Size dim_visible by n_hidden by dim_hidden
            return np.choose(xi.reshape((-1, 1, 1)), logp).transpose((1, 0, 2))

        else:
            print('Marginal description "%s" not implemented.' % self.marginal_description)
            sys.exit()

    def estimate_parameters(self, xi, p_y_given_x):
        if self.marginal_description == 'gaussian':
            n_obs = np.sum(p_y_given_x, axis=1).clip(0.1)  # m, k
            mean_ml = np.einsum('i,jik->jk', xi, p_y_given_x, optimize=False) / n_obs  # ML estimate of mean of Xi
            sig_ml = np.einsum('jik,jik->jk', (xi[np.newaxis, :, np.newaxis] - mean_ml[:, np.newaxis, :])**2, p_y_given_x, optimize=False) / (n_obs - 1).clip(0.01)  # UB estimate of sigma^2(variance)

            if not self.smooth_marginals:
                return np.array([mean_ml, sig_ml])  # FOR EACH Y_j = k !!
            else:  # mu = lam mu_ml + 1-lam mu0 for lam minimizing KL divergence risk
                mean0 = np.mean(xi)
                sig0 = np.sum((xi - mean0)**2) / (len(xi) - 1)
                m1, m2, se1, se2 = self.estimate_se(xi, p_y_given_x, n_obs)
                d1 = mean_ml - m1
                d2 = sig_ml - m2
                lam = d1**2 / (d1**2 + se1**2)
                gam = d2**2 / (d2**2 + se2**2)
                lam, gam = np.where(np.isfinite(lam), lam, 0.5), np.where(np.isfinite(gam), gam, 0.5)
                # lam2 = 1. - 1. / (1. + n_obs)  # Constant pseudo-count, doesn't work as well.
                # gam2 = 1. - 1. / (1. + n_obs)
                mean_prime = lam * mean_ml + (1. - lam) * mean0
                sig_prime = gam * sig_ml + (1. - gam) * sig0
                return np.array([mean_prime, sig_prime])  # FOR EACH Y_j = k !!

        elif self.marginal_description == 'discrete':
            # Discrete data: should be non-negative integers starting at 0: 0,...k
            x_select = (xi == np.arange(self.dim_visible)[:, np.newaxis])  # dim_v by ns
            prior = np.mean(x_select, axis=1).reshape((-1, 1, 1))  # dim_v, 1, 1
            n_obs = np.sum(p_y_given_x, axis=1)  # m, k
            counts = np.dot(x_select, p_y_given_x)  # dim_v, m, k
            p = counts + 0.001  # Tiny smoothing to avoid numerical errors
            p /= p.sum(axis=0, keepdims=True)
            if self.smooth_marginals:  # Shrinkage interpreted as hypothesis testing...
                G_stat = 2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0)
                G0 = self.estimate_sig(x_select, p_y_given_x, n_obs, prior)
                z = 1
                lam = G_stat**z / (G_stat**z + G0**z)
                lam = np.where(np.isnan(lam), 0.5, lam)
                p = (1 - lam) * prior + lam * p
            return np.log(p)

        else:
            print('Marginal description "%s" not implemented.' % self.marginal_description)
            sys.exit()

    def estimate_se(self, xi, p_y_given_x, n_obs):
        # Get a bootstrap estimate of mean and standard error for estimating mu and sig^2 given | Y_j=k  (under null)
        # x_copy = np.hstack([np.random.choice(xi, size=(len(xi), 1), replace=False) for _ in range(20)])
        x_copy = np.random.choice(xi, size=(len(xi), 20), replace=True)  # w/o replacement leads to...higher s.e. and more smoothing.
        m, n, k = p_y_given_x.shape
        mean_ml = np.einsum('il,jik->jkl', x_copy, p_y_given_x, optimize=False) / n_obs[..., np.newaxis]  # ML estimate
        sig_ml = np.einsum('jikl,jik->jkl', (x_copy.reshape((1, n, 1, 20)) - mean_ml.reshape((m, 1, k, 20)))**2, p_y_given_x, optimize=False) / (n_obs[..., np.newaxis] - 1).clip(0.01) # ML estimate
        m1 = np.mean(mean_ml, axis=2)
        m2 = np.mean(sig_ml, axis=2)
        se1 = np.sqrt(np.sum((mean_ml - m1[..., np.newaxis])**2, axis=2) / 19.)
        se2 = np.sqrt(np.sum((sig_ml - m2[..., np.newaxis])**2, axis=2) / 19.)
        return m1, m2, se1, se2

    def estimate_sig(self, x_select, p_y_given_x, n_obs, prior):
        # Permute p_y_given_x, est mean Gs
        # TODO: This should be done using sampling with replacement instead of permutation.
        Gs = []
        for i in range(20):
            order = np.random.permutation(p_y_given_x.shape[1])
            counts = np.dot(x_select, p_y_given_x[:, order, :])  # dim_v, m, k
            Gs.append(2 * np.sum(np.where(counts > 0, counts * (np.log(counts) - np.log(n_obs * prior)), 0), axis=0))
        return np.mean(Gs, axis=0)


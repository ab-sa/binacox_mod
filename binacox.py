import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
#from tick.preprocessing.features_binarizer import FeaturesBinarizer
#from tick.survival import CoxRegression
from lifelines.utils import concordance_index
import statsmodels.api as sm
import pylab as pl
import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from lifelines import CoxPHFitter
import numpy as np
import scipy.stats as stats

# Custom implementation for SimuCoxRegWithCutPoints
# This function simulates covariates and survival times for a Cox proportional hazards model

def simu_cox_reg_with_cutpoints(n_samples=100, n_features=10, cutpoints=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate covariates (features)
    X = np.random.randn(n_samples, n_features)
    
    # Apply cutpoints if provided
    if cutpoints is not None:
        for feature_idx, cutpoint in cutpoints.items():
            X[:, feature_idx] = np.where(X[:, feature_idx] > cutpoint, 1, 0)
    
    # Generate baseline hazard and survival times
    baseline_hazard = 0.01  # Assume a constant baseline hazard for simplicity
    linear_predictor = np.dot(X, np.random.uniform(-1, 1, n_features))
    hazard = baseline_hazard * np.exp(linear_predictor)
    
    # Generate survival times using an exponential distribution
    survival_times = np.random.exponential(1 / hazard)
    
    # Generate censoring times and apply censoring
    censoring_times = np.random.uniform(0, np.max(survival_times), n_samples)
    observed_times = np.minimum(survival_times, censoring_times)
    event_observed = survival_times <= censoring_times
    
    return X, observed_times, event_observed

def custom_binarizer(X, n_bins=50, strategy='uniform'):
    # Step 1: Discretize the features using KBinsDiscretizer
    binarizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    X_binned = binarizer.fit_transform(X)
    
    # Step 2: One-hot encode the binned features
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    X_bin = one_hot_encoder.fit_transform(X_binned)
    
    # Step 3: Extract bin boundaries
    bins_boundaries = binarizer.bin_edges_
    
    # Step 4: Create mapper and feature_type (assuming all features are continuous for simplicity)
    mapper = {i: list(range(len(one_hot_encoder.categories_[i]))) for i in range(len(one_hot_encoder.categories_))}
    feature_type = {i: 'continuous' for i in range(X.shape[1])}
    
    # Step 5: Calculate blocks_start and blocks_length
    blocks_start = [0]
    blocks_length = []
    for i in range(len(one_hot_encoder.categories_)):
        length = len(one_hot_encoder.categories_[i])
        blocks_length.append(length)
        if i > 0:
            blocks_start.append(blocks_start[-1] + blocks_length[-2])
    
    return {
        'X_bin': X_bin,
        'one_hot_encoder': one_hot_encoder,
        'boundaries': bins_boundaries,
        'mapper': mapper,
        'feature_type': feature_type,
        'blocks_start': blocks_start,
        'blocks_length': blocks_length
    }


# Custom implementation for ProxBinarsity
# This class applies total-variation regularization followed by centering within sub-blocks
def prox_binarsity(beta, strength, blocks_start, blocks_length, positive=False):
    n_blocks = len(blocks_start)
    new_beta = beta.copy()
    for i in range(n_blocks):
        start = blocks_start[i]
        length = blocks_length[i]
        block = beta[start:start + length]
        # Apply total variation denoising (TV regularization)
        block_diff = np.diff(block)
        tv_penalty = strength * np.sign(block_diff)
        block[:-1] -= tv_penalty
        # Centering within sub-blocks
        block_mean = np.mean(block)
        block -= block_mean
        # Ensure non-negative entries if positive=True
        if positive:
            block = np.maximum(block, 0)
        new_beta[start:start + length] = block
    return new_beta

# Update any instances of functions and classes from tick to use the new library imports.
# Ensure that the inputs/outputs of the replacement functions are compatible with the rest of the binacox code.

# Step 3: Modify binacox code according to the alternatives
# Modify the relevant functions in binacox to use KBinsDiscretizer, CoxPHFitter, or other suitable alternatives.
# Make sure to adjust the code where these classes or functions are used, such as data preprocessing, model fitting, or simulation.

# Custom Cox regression with binarsity penalty
class CoxRegression:
    def __init__(self, penalty='binarsity', tol=1e-5, verbose=False, max_iter=100, step=0.3,
                 blocks_start=None, blocks_length=None, warm_start=True):
        self.penalty = penalty
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.step = step
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length
        self.warm_start = warm_start
        self.beta = None

    def fit(self, X, Y, delta):
        # Initialize coefficients
        n_features = X.shape[1]
        self.beta = np.zeros(n_features)
        for iteration in range(self.max_iter):
            # Compute gradient of the negative log partial likelihood
            risk_scores = np.exp(np.dot(X, self.beta))
            partial_likelihood_gradient = -np.dot(X.T, delta - (risk_scores / np.sum(risk_scores)))
            # Apply proximal operator if penalty is 'binarsity'
            if self.penalty == 'binarsity':
                self.beta -= self.step * partial_likelihood_gradient
                self.beta = prox_binarsity(self.beta, self.step, self.blocks_start, self.blocks_length)
            else:
                self.beta -= self.step * partial_likelihood_gradient
            # Check for convergence
            if np.linalg.norm(partial_likelihood_gradient) < self.tol:
                if self.verbose:
                    print(f'Convergence reached at iteration {iteration}')
                break




def compute_score(features, features_binarized, times, censoring,
                  blocks_start, blocks_length, boundaries, C=10, n_folds=10,
                  features_names=None, shuffle=True, n_jobs=1, verbose=False,
                  validation_data=None):
    scores = cross_val_score(features, features_binarized, times,
                             censoring, blocks_start, blocks_length, boundaries,
                             n_folds=n_folds, shuffle=shuffle, C=C,
                             features_names=features_names, n_jobs=n_jobs,
                             verbose=verbose, validation_data=validation_data)
    scores_test = scores[:, 0]
    scores_validation = scores[:, 1]
    if validation_data is not None:
        scores_validation_mean = scores_validation.mean()
        scores_validation_std = scores_validation.std()
    else:
        scores_validation_mean, scores_validation_std = None, None

    scores_mean = scores_test.mean()
    scores_std = scores_test.std()
    if verbose:
        print("\nscore %0.3f (+/- %0.3f)" % (scores_mean, scores_std))
    scores = [scores_mean, scores_std, scores_validation_mean,
              scores_validation_std]
    return scores


def cross_val_score(features, features_binarized, times, censoring,
                    blocks_start, blocks_length, boundaries, n_folds, shuffle,
                    C, features_names, n_jobs, verbose, validation_data):
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    cv_iter = list(cv.split(features))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    scores = parallel(
        delayed(fit_and_score)(features, features_binarized, times,
                               censoring, blocks_start, blocks_length,
                               boundaries, features_names, idx_train, idx_test,
                               validation_data, C)
        for (idx_train, idx_test) in cv_iter)
    return np.array(scores)


def fit_and_score(features, features_bin, times, censoring,
                  blocks_start, blocks_length, boundaries, features_names,
                  idx_train, idx_test, validation_data, C):
    if features_names is None:
        features_names = [str(j) for j in range(features.shape[1])]
    X_train, X_test = features_bin[idx_train], features_bin[idx_test]
    Y_train, Y_test = times[idx_train], times[idx_test]
    delta_train, delta_test = censoring[idx_train], censoring[idx_test]

    learner = CoxRegression(penalty='binarsity', tol=1e-5,
                            verbose=False, max_iter=100, step=0.3,
                            blocks_start=blocks_start,
                            blocks_length=blocks_length,
                            warm_start=True)
    learner._solver_obj.linesearch = False
    learner.C = C
    learner.fit(X_train, Y_train, delta_train)
    coeffs = learner.coeffs

    cut_points_estimates = {}
    for j, start in enumerate(blocks_start):
        coeffs_j = coeffs[start:start + blocks_length[j]]
        all_zeros = not np.any(coeffs_j)
        if all_zeros:
            cut_points_estimate_j = np.array([-np.inf, np.inf])
        else:
            groups_j = get_groups(coeffs_j)
            jump_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1
            if jump_j.size == 0:
                cut_points_estimate_j = np.array([-np.inf, np.inf])
            else:
                cut_points_estimate_j = boundaries[features_names[j]][
                    jump_j]
                if cut_points_estimate_j[0] != -np.inf:
                    cut_points_estimate_j = np.insert(cut_points_estimate_j,
                                                      0, -np.inf)
                if cut_points_estimate_j[-1] != np.inf:
                    cut_points_estimate_j = np.append(cut_points_estimate_j,
                                                      np.inf)
        cut_points_estimates[features_names[j]] = cut_points_estimate_j
    binarizer = FeaturesBinarizer(method='given',
                                  bins_boundaries=cut_points_estimates)
    binarized_features = binarizer.fit_transform(features)
    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length
    X_bin_train = binarized_features[idx_train]
    X_bin_test = binarized_features[idx_test]
    learner_ = CoxRegression(penalty='binarsity', tol=1e-5,
                             verbose=False, max_iter=100, step=0.3,
                             blocks_start=blocks_start,
                             blocks_length=blocks_length,
                             warm_start=True, C=1e10)
    learner_._solver_obj.linesearch = False
    learner_.fit(X_bin_train, Y_train, delta_train)
    score = learner_.score(X_bin_test, Y_test, delta_test)

    if validation_data is not None:
        X_validation = validation_data[0]
        X_bin_validation = binarizer.fit_transform(X_validation)
        Y_validation = validation_data[1]
        delta_validation = validation_data[2]
        score_validation = learner_.score(X_bin_validation, Y_validation,
                                          delta_validation)
    else:
        score_validation = None

    return score, score_validation


def get_groups(coeffs):
    n_coeffs = len(coeffs)
    jumps = np.where(coeffs[1:] - coeffs[:-1] != 0)[0] + 1
    jumps = np.insert(jumps, 0, 0)
    jumps = np.append(jumps, n_coeffs)
    groups = np.zeros(n_coeffs)
    for i in range(len(jumps) - 1):
        groups[jumps[i]:jumps[i + 1]] = i
        if jumps[i + 1] - jumps[i] <= 2:
            if i == 0:
                groups[jumps[i]:jumps[i + 1]] = 1
            elif i == len(jumps) - 2:
                groups[jumps[i]:jumps[i + 1]] = groups[jumps[i - 1]]
            else:
                coeff_value = coeffs[jumps[i]]
                group_before = groups[jumps[i - 1]]
                coeff_value_before = coeffs[
                    np.nonzero(groups == group_before)[0][0]]
                try:
                    k = 0
                    while coeffs[jumps[i + 1] + k] != coeffs[
                                        jumps[i + 1] + k + 1]:
                        k += 1
                    coeff_value_after = coeffs[jumps[i + 1] + k]
                except:
                    coeff_value_after = coeffs[jumps[i + 1]]
                if np.abs(coeff_value_before - coeff_value) < np.abs(
                                coeff_value_after - coeff_value):
                    groups[np.where(groups == i)] = group_before
                else:
                    groups[np.where(groups == i)] = i + 1
    return groups.astype(int)


def get_m_1(cut_points_estimates, cut_points, S):
    m_1, d = 0, 0
    n_features = len(cut_points)
    for j in set(range(n_features)) - set(S):
        mu_star_j = cut_points[str(j)][1:-1]
        hat_mu_star_j = cut_points_estimates[str(j)][1:-1]
        if len(hat_mu_star_j) > 0:
            d += 1
            m_1 += get_H(mu_star_j, hat_mu_star_j)
    if d == 0:
        m_1 = np.nan
    else:
        m_1 *= (1 / d)
    return m_1


def get_H(A, B):
    return max(get_E(A, B), get_E(B, A))


def get_E(A, B):
    return max([min([abs(a - b) for a in A]) for b in B])


def get_m_2(hat_K_star, S):
    return (1 / len(S)) * hat_K_star[S].sum()


def plot_screening(screening_strategy, screening_marker, cancer, P):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    alpha = .8
    lw = 2
    label = 'Selected'
    n_features = len(screening_marker)
    ax.plot(range(P), screening_marker[:P], 'r',
            lw=lw, alpha=alpha, label=label)
    label = 'Rejected'
    ax.plot(range(P, n_features), screening_marker[P:],
            'b', lw=lw, alpha=alpha, label=label)
    pl.legend(fontsize=18)
    pl.xlabel(r'$j$', fontsize=25)
    pl.tick_params(axis='x', which='both', top='off')
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.title("%s screening on %s" % (screening_strategy, cancer),
             fontsize=20)
    pl.tight_layout()
    pl.show()


def get_p_values_j(feature, mu_k, times, censoring, values_to_test, epsilon):
    if values_to_test is None:
        p1 = np.percentile(feature, epsilon)
        p2 = np.percentile(feature, 100 - epsilon)
        values_to_test = mu_k[np.where((mu_k <= p2) & (mu_k >= p1))]
    p_values, t_values = [], []
    for val in values_to_test:
        feature_bin = feature <= val
        mod = sm.PHReg(endog=times, status=censoring,
                       exog=feature_bin.astype(int), ties="efron")
        fitted_model = mod.fit()
        p_values.append(fitted_model.pvalues[0])
        t_values.append(fitted_model.tvalues[0])
    p_values = pd.DataFrame({'values_to_test': values_to_test,
                             'p_values': p_values,
                             't_values': t_values})
    p_values.sort_values('values_to_test', inplace=True)
    return p_values


def multiple_testing(X, boundaries, Y, delta, values_to_test=None,
                     features_names=None, epsilon=5):
    if values_to_test is None:
        values_to_test = X.shape[1] * [None]
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    X = np.array(X)
    result = Parallel(n_jobs=5)(
        delayed(get_p_values_j)(X[:, j],
                                boundaries[features_names[j]].copy()[1:-1], Y,
                                delta, values_to_test[j], epsilon=epsilon)
        for j in range(X.shape[1]))
    return result


def t_ij(i, j, n):
    return (1 - i * (n - j) / ((n - i) * j)) ** .5


def d_ij(i, j, z, n):
    return (2 / np.pi) ** .5 * norm.pdf(z) * (
        t_ij(i, j, n) - (z ** 2 / 4 - 1) * t_ij(i, j, n) ** 3 / 6)


def p_value_cut(p_values, values_to_test, feature, epsilon=5):
    n_tested = p_values.size
    p_value_min = np.min(p_values)
    l = np.zeros(n_tested)
    l[-1] = n_tested
    d = np.zeros(n_tested - 1)
    z = norm.ppf(1 - p_value_min / 2)
    values_to_test_sorted = np.sort(values_to_test)

    epsilon /= 100
    p_corr_1 = norm.pdf(1 - p_value_min / 2) * (z - 1 / z) * np.log(
        (1 - epsilon) ** 2 / epsilon ** 2) + 4 * norm.pdf(z) / z

    for i in np.arange(n_tested - 1):
        l[i] = np.count_nonzero(feature <= values_to_test_sorted[i])
        if i >= 1:
            d[i - 1] = d_ij(l[i - 1], l[i], z, feature.shape[0])
    p_corr_2 = p_value_min + np.sum(d)

    p_value_min_corrected = np.min((p_corr_1, p_corr_2, 1))
    if np.isnan(p_value_min_corrected) or np.isinf(p_value_min_corrected):
        p_value_min_corrected = p_value_min
    return p_value_min_corrected


def multiple_testing_perm(n_samples, X, boundaries, Y, delta, values_to_test_init,
                     features_names, epsilon):
    np.random.seed()
    perm = np.random.choice(n_samples, size=n_samples, replace=True)
    multiple_testing_rslt = multiple_testing(X[perm], boundaries, Y[perm],
                                   delta[perm], values_to_test_init,
                                   features_names=features_names,
                                   epsilon=epsilon)
    return multiple_testing_rslt


def bootstrap_cut_max_t(X, boundaries, Y, delta, multiple_testing_rslt, B=10,
                        features_names=None, epsilon=5):
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    n_samples, n_features = X.shape
    t_values_init, values_to_test_init, t_values_B = [], [], []
    for j in range(n_features):
        t_values_init.append(multiple_testing_rslt[j].t_values)
        values_to_test_j = multiple_testing_rslt[j].values_to_test
        values_to_test_init.append(values_to_test_j)
        n_tested_j = values_to_test_j.size
        t_values_B.append(pd.DataFrame(np.zeros((B, n_tested_j))))

    result = Parallel(n_jobs=10)(
        delayed(multiple_testing_perm)(n_samples, X, boundaries, Y, delta,
                                  values_to_test_init, features_names, epsilon)
        for _ in np.arange(B))

    for b in np.arange(B):
        for j in range(n_features):
            t_values_B[j].ix[b, :] = result[b][j].t_values

    adjusted_p_values = []
    for j in range(n_features):
        sd = t_values_B[j].std(0)
        sd[sd < 1] = 1
        mean = t_values_B[j].mean(0)
        t_val_B_H0_j = (t_values_B[j] - mean) / sd
        maxT_j = t_val_B_H0_j.abs().max(1)
        adjusted_p_values.append(
            [(maxT_j > np.abs(t_k)).mean() for t_k in t_values_init[j]])
    return adjusted_p_values


def refit_and_predict(cut_points_estimates, X_train, X_test, Y_train,
                      delta_train, Y_test, delta_test):

    binarizer = FeaturesBinarizer(method='given',
                                  bins_boundaries=cut_points_estimates,
                                  remove_first=True)
    binarizer.fit(pd.concat([X_train, X_test]))
    X_bin_train = binarizer.transform(X_train)
    X_bin_test = binarizer.transform(X_test)

    learner = CoxRegression(penalty='none', tol=1e-5,
                            solver='agd', verbose=False,
                            max_iter=100, step=0.3,
                            warm_start=True)
    learner._solver_obj.linesearch = False
    learner.fit(X_bin_train, Y_train, delta_train)
    coeffs = learner.coeffs
    marker = X_bin_test.dot(coeffs)
    lp_train = X_bin_train.dot(coeffs)
    c_index = concordance_index(Y_test, marker, delta_test)
    c_index = max(c_index, 1 - c_index)

    return c_index, marker, lp_train

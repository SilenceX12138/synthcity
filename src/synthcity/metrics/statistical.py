# stdlib
from typing import Dict, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from copulas.univariate.base import Univariate
from dython.nominal import associations
from geomloss import SamplesLoss
from pydantic import validate_arguments
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn import metrics

# synthcity absolute
from synthcity.metrics._utils import get_freq


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_inv_kl_divergence(X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> float:
    """Returns the average inverse of the Kullback–Leibler Divergence metric.

    Score:
        0: the datasets are from different distributions.
        1: the datasets are from the same distribution.
    """
    freqs = get_freq(X_gt, X_syn)
    res = []
    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_kolmogorov_smirnov_test(X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> float:
    """Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    res = []
    for col in X_gt.columns:
        statistic, _ = ks_2samp(X_gt[col], X_syn[col])
        res.append(1 - statistic)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_chi_squared_test(X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> float:
    """Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.

    Score:
        0: the distributions are different
        1: the distributions are identical.
    """

    res = []
    freqs = get_freq(X_gt, X_syn)

    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        try:
            _, pvalue = chisquare(gt_freq, synth_freq)
        except BaseException:
            pvalue = 0

        res.append(pvalue)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_maximum_mean_discrepancy(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    kernel: str = "rbf",
) -> float:
    """Empirical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.

    Args:
        kernel: "rbf", "linear" or "polynomial"

    Score:
        0: The distributions are the same.
        1: The distributions are totally different.
    """
    if kernel == "linear":
        """
        MMD using linear kernel (i.e., k(x,y) = <x,y>)
        """
        delta_df = X_gt.mean(axis=0) - X_syn.mean(axis=0)
        delta = delta_df.values

        return delta.dot(delta.T)
    elif kernel == "rbf":
        """
        MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        """
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(X_gt, X_gt, gamma)
        YY = metrics.pairwise.rbf_kernel(X_syn, X_syn, gamma)
        XY = metrics.pairwise.rbf_kernel(X_gt, X_syn, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    elif kernel == "polynomial":
        """
        MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        """
        degree = 2
        gamma = 1
        coef0 = 0
        XX = metrics.pairwise.polynomial_kernel(X_gt, X_gt, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X_gt, X_syn, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        raise ValueError(f"Unsupported kernel {kernel}")


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_inv_cdf_distance(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    p: int = 2,
) -> float:
    """Evaluate the distance between continuous features."""
    dist = 0
    for col in X_syn.columns:
        if len(X_syn[col].unique()) < 20:
            continue
        syn_col = X_syn[col]
        gt_col = X_gt[col]

        predictor = Univariate()
        predictor.fit(syn_col)

        syn_percentiles = predictor.cdf(np.array(syn_col))
        gt_percentiles = predictor.cdf(np.array(gt_col))
        dist += np.mean(abs(syn_percentiles - gt_percentiles[1]) ** p)

    return dist


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_avg_jensenshannon_stats(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    normalize: bool = True,
    bins: int = 10,
) -> Tuple[Dict, Dict, Dict]:
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""

    stats_gt = {}
    stats_syn = {}
    stats_ = {}

    for col in X_gt.columns:
        local_bins = min(bins, len(X_gt[col].unique()))
        X_gt_bin, gt_bins = pd.cut(X_gt[col], bins=local_bins, retbins=True)
        X_syn_bin = pd.cut(X_syn[col], bins=gt_bins)
        stats_gt[col], stats_syn[col] = X_gt_bin.value_counts(
            dropna=False, normalize=normalize
        ).align(
            X_syn_bin.value_counts(dropna=False, normalize=normalize),
            join="outer",
            axis=0,
            fill_value=0,
        )
        stats_[col] = jensenshannon(stats_gt[col], stats_syn[col])

    return stats_, stats_gt, stats_syn


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_avg_jensenshannon_distance(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    normalize: bool = True,
) -> float:
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""
    stats_, _, _ = evaluate_avg_jensenshannon_stats(X_gt, X_syn)

    return sum(stats_.values()) / len(stats_.keys())


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_feature_correlation_stats(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    nom_nom_assoc: str = "theil",
    nominal_columns: str = "auto",
) -> Tuple[Dict, Dict]:
    """Evaluate the correlation/strength-of-association of features in data-set with both categorical and continuous features using: * Pearson's R for continuous-continuous cases ** Cramer's V or Theil's U for categorical-categorical cases."""
    stats_gt = associations(
        X_gt,
        nom_nom_assoc=nom_nom_assoc,
        nominal_columns=nominal_columns,
        nan_replace_value="nan",
        compute_only=True,
    )["corr"]
    stats_syn = associations(
        X_syn,
        nom_nom_assoc=nom_nom_assoc,
        nominal_columns=nominal_columns,
        nan_replace_value="nan",
        compute_only=True,
    )["corr"]

    return stats_gt, stats_syn


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_feature_correlation(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    nom_nom_assoc: str = "theil",
    nominal_columns: str = "auto",
) -> float:
    """Evaluate the correlation/strength-of-association of features in data-set with both categorical and continuous features using: * Pearson's R for continuous-continuous cases ** Cramer's V or Theil's U for categorical-categorical cases."""
    stats_gt, stats_syn = evaluate_feature_correlation_stats(X_gt, X_syn)

    return np.linalg.norm(stats_gt - stats_syn, "fro")


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_wasserstein_distance(
    X: pd.DataFrame,
    X_syn: pd.DataFrame,
) -> float:
    """Compare Wasserstein distance between original data and synthetic data.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        WD_value: Wasserstein distance
    """
    X_ten = torch.from_numpy(X.values)
    Xsyn_ten = torch.from_numpy(X_syn.values)
    OT_solver = SamplesLoss(loss="sinkhorn")

    return OT_solver(X_ten, Xsyn_ten).cpu().numpy()
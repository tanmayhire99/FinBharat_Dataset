"""
Statistical significance testing for FinBharat evaluation results.

- Bootstrap 95% confidence intervals (1000 iterations)
- Paired bootstrap p-values for model comparisons
- All results from two EvalResult lists (same QA, different models/regimes)
"""
import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class BootstrapCI:
    mean: float
    lower: float   # 2.5th percentile
    upper: float   # 97.5th percentile
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class PairedBootstrapResult:
    delta: float          # mean_a - mean_b
    p_value: float        # fraction of bootstrap samples where delta_b >= delta_a
    significant: bool     # p_value < 0.05
    ci_a: BootstrapCI
    ci_b: BootstrapCI


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap 95% CI for a list of per-sample scores."""
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return BootstrapCI(mean=0.0, lower=0.0, upper=0.0, n=0)

    mean = sum(scores) / n
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    lower = bootstrap_means[int(0.025 * n_bootstrap)]
    upper = bootstrap_means[int(0.975 * n_bootstrap)]
    return BootstrapCI(mean=round(mean, 4), lower=round(lower, 4), upper=round(upper, 4), n=n)


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PairedBootstrapResult:
    """
    Paired bootstrap test: is system A significantly better than system B?

    Returns p-value = P(delta_bootstrap >= delta_observed) under H0: A == B.
    Significant if p_value < 0.05.

    Reference: Berg-Kirkpatrick et al. (2012), Dror et al. (2018 EMNLP).
    """
    assert len(scores_a) == len(scores_b), "Score lists must be same length"
    n = len(scores_a)
    rng = random.Random(seed)

    delta_obs = sum(scores_a) / n - sum(scores_b) / n
    ci_a = bootstrap_ci(scores_a, n_bootstrap, seed)
    ci_b = bootstrap_ci(scores_b, n_bootstrap, seed)

    # Paired bootstrap: resample indices, count how often delta >= 2 * delta_obs
    count = 0
    for _ in range(n_bootstrap):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        d = sum(scores_a[i] - scores_b[i] for i in idx) / n
        # Two-sided correction: p = P(|delta*| >= |delta_obs|)
        if abs(d - delta_obs) >= abs(delta_obs):
            count += 1

    p_value = round(count / n_bootstrap, 4)
    return PairedBootstrapResult(
        delta=round(delta_obs, 4),
        p_value=p_value,
        significant=p_value < 0.05,
        ci_a=ci_a,
        ci_b=ci_b,
    )


def compute_significance_table(
    results_a: list,   # list[EvalResult]
    results_b: list,   # list[EvalResult]
    metric_getters: dict[str, Callable] | None = None,
) -> dict[str, PairedBootstrapResult]:
    """
    Compute paired bootstrap significance for all key metrics between two runs.

    Args:
        results_a: EvalResult list for system A (e.g. llama3.3-70b)
        results_b: EvalResult list for system B (e.g. llama3.1-8b)
        metric_getters: {metric_name: lambda r: score}. Defaults to core metrics.

    Returns:
        {metric_name: PairedBootstrapResult}
    """
    if metric_getters is None:
        metric_getters = {
            "exact_match":         lambda r: float(r.exact_match),
            "relaxed_em":          lambda r: float(r.relaxed_em),
            "token_f1":            lambda r: float(r.token_f1),
            "rouge_l":             lambda r: float(r.rouge_l),
            "bertscore_f1":        lambda r: float(r.bertscore_f1),
            "num_exact":           lambda r: float(r.num_exact),
            "num_f1":              lambda r: float(r.num_f1),
            "entailment_ratio":    lambda r: float(r.entailment_ratio),
        }

    # Align by question_id
    id_to_a = {r.question_id: r for r in results_a}
    id_to_b = {r.question_id: r for r in results_b}
    shared = [qid for qid in id_to_a if qid in id_to_b]

    if not shared:
        raise ValueError("No shared question_ids between result sets — cannot run paired test.")

    out = {}
    for metric, getter in metric_getters.items():
        sa = [getter(id_to_a[qid]) for qid in shared]
        sb = [getter(id_to_b[qid]) for qid in shared]
        out[metric] = paired_bootstrap_test(sa, sb)
    return out


def add_ci_to_aggregate(agg: dict, eval_results: list, n_bootstrap: int = 1000) -> dict:
    """
    Enrich an aggregate dict with bootstrap CIs for all numeric metrics.
    Adds keys like 'exact_match_ci', 'token_f1_ci', etc.
    """
    metric_map = {
        "exact_match":      lambda r: float(r.exact_match),
        "relaxed_em":       lambda r: float(r.relaxed_em),
        "token_f1":         lambda r: float(r.token_f1),
        "rouge_l":          lambda r: float(r.rouge_l),
        "bertscore_f1":     lambda r: float(r.bertscore_f1),
        "num_exact":        lambda r: float(r.num_exact),
        "num_f1":           lambda r: float(r.num_f1),
        "entailment_ratio": lambda r: float(r.entailment_ratio),
    }
    for metric, getter in metric_map.items():
        scores = [getter(r) for r in eval_results if not r.error]
        ci = bootstrap_ci(scores, n_bootstrap=n_bootstrap)
        agg[f"{metric}_ci"] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper, "n": ci.n}
    return agg

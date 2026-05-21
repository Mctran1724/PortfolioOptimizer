"""Analytics subpackage exposing traditional and tail risk metrics, factor regression, and fixed income formulas."""

from portfolio_optimizer.analytics.risk_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_treynor_ratio,
    calculate_information_ratio
)

from portfolio_optimizer.analytics.tail_risk import (
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_ulcer_index
)

from portfolio_optimizer.analytics.factor_models import (
    run_fama_french_regression,
    marcenko_pastur_filter
)

from portfolio_optimizer.analytics.fixed_income import (
    calculate_bond_duration_and_convexity,
    calculate_breakeven_inflation,
    simulate_svf_crediting_rate
)

from portfolio_optimizer.analytics.annuities import (
    calculate_indexed_annuity_payoff,
    monte_carlo_annuity_pricing
)

from portfolio_optimizer.analytics.sector_analysis import RandomForestSectorRotator

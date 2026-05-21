# Investment Portfolio Optimization Framework - Project Structure

## Overview
This document outlines the proposed project structure for a scalable quantitative portfolio optimization framework that enables:
1. Easy addition of new metrics and optimization techniques
2. Modular design for maintainability
3. Streamlit dashboard for visualization
4. Integration with various data sources (yfinance, fredapi, FinanceToolkit)
5. Advanced analytics including factor models, tail risk metrics, and optimization algorithms

## Proposed Directory Structure
```
investmentportfolio/
│
├── data/                           # Data storage (cached data, raw data)
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Processed/cleaned data
│   └── cache/                      # Cached API responses
│
├── portfolio_optimizer/            # Core optimization package
│   ├── __init__.py
│   │
│   ├── data/                       # Data ingestion modules
│   │   ├── __init__.py
│   │   ├── ingestion.py            # Existing data ingestion (yfinance, fredapi, financetoolkit)
│   │   ├── cache.py                # Data caching layer
│   │   └── validators.py           # Data validation utilities
│   │
│   ├── analytics/                  # Analytics and metrics calculation
│   │   ├── __init__.py
│   │   ├── risk_metrics.py         # Sharpe, Sortino, Treynor, Information ratios
│   │   ├── tail_risk.py            # VaR, CVaR, Maximum Drawdown, Calmar, Ulcer Index
│   │   ├── factor_models.py        # Fama-French, CAPM, factor regression
│   │   ├── sector_analysis.py      # Sector rotation, correlation analysis
│   │   └── fixed_income.py         # Duration, convexity, TIPS analysis
│   │
│   ├── optimization/               # Portfolio optimization engines
│   │   ├── __init__.py
│   │   ├── base_optimizer.py       # Abstract base class for optimizers
│   │   ├── traditional.py          # Mean-Variance, Risk Parity
│   │   ├── hierarchical.py         # HRP, HERC
│   │   ├── black_litterman.py      # Black-Litterman model
│   │   ├── kelly_criterion.py      # Kelly criterion optimization
│   │   └── shortfall_minimization.py # CVaR/CDaR minimization
│   │
│   ├── constraints/                # Optimization constraints
│   │   ├── __init__.py
│   │   ├── linear_constraints.py   # Linear constraints (beta limits, factor exposures)
│   │   ├── risk_constraints.py     # Risk-based constraints (CVaR limits, drawdown limits)
│   │   └── factor_constraints.py   # Factor exposure constraints
│   │
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py              # General helper functions
│   │   ├── constants.py            # Constants and configuration
│   │   └── logging.py              # Logging configuration
│   │
│   └── models/                     # Data models and schemas
│       ├── __init__.py
│       ├── portfolio.py            # Portfolio data structure
│       ├── asset.py                # Asset definition
│       └── optimization_result.py  # Results from optimization
│
├── dashboard/                      # Streamlit dashboard components
│   ├── __init__.py
│   ├── components/                 # Reusable dashboard components
│   │   ├── __init__.py
│   │   ├── metrics_display.py      # Metrics visualization components
│   │   ├── portfolio_viz.py        # Portfolio allocation visualizations
│   │   ├── risk_analysis.py        # Risk analysis visualizations
│   │   └── factor_analysis.py      # Factor exposure visualizations
│   │
│   ├── pages/                      # Streamlit page definitions
│   │   ├── __init__.py
│   │   ├── 01_Data_Ingestion.py    # Data input and ingestion page
│   │   ├── 02_Analytics.py         # Analytics and metrics page
│   │   ├── 03_Optimization.py      # Portfolio optimization page
│   │   ├── 04_Backtesting.py       # Backtesting and analysis page
│   │   └── 05_Reports.py           # Reports and exports page
│   │
│   ├── utils/                      # Dashboard utilities
│   │   ├── __init__.py
│   │   ├── state_management.py     # Streamlit state management
│   │   └── formatting.py           # Number formatting utilities
│   │
│   └── app.py                      # Main Streamlit application
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── fixtures/                   # Test fixtures
│
├── scripts/                        # Utility scripts
│   ├── data_download.py            # Script for batch data downloading
│   ├── run_optimization.py         # Script for running optimization from CLI
│   └── generate_report.py          # Script for generating reports
│
├── config/                         # Configuration files
│   ├── default_config.yaml         # Default configuration
│   └── environments/               # Environment-specific configs
│       ├── development.yaml
│       ├── production.yaml
│       └── testing.yaml
│
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── pyproject.toml                  # Existing project configuration
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

## Key Design Principles

### 1. Modularity and Separation of Concerns
- Each major functionality area (data ingestion, analytics, optimization) is separated into distinct modules
- Clear interfaces between components using abstract base classes and dependency injection
- Easy to extend with new metrics, optimization techniques, or data sources

### 2. Scalability for New Metrics
- Analytics module is organized by metric type (risk_metrics, tail_risk, factor_models, etc.)
- Adding new metrics requires creating a new file in the appropriate subdirectory
- Metrics follow a consistent interface for easy integration

### 3. Extensible Optimization Framework
- Base optimizer class defines common interface
- Each optimization technique (Traditional, Hierarchical, Black-Litterman, etc.) implements the base interface
- Constraints are modular and can be combined flexibly

### 4. Dashboard-Focused Design
- Streamlit components are organized for reuse across pages
- Clear separation between data processing and visualization
- State management utilities to handle Streamlit's reactive nature

### 5. Data Management
- Separation of raw, processed, and cached data
- Caching layer to minimize API calls
- Data validation to ensure quality inputs

## Implementation Priority

### Phase 1: Core Framework Setup
1. Create the directory structure outlined above
2. Migrate existing ingestion.py and analytics.py to new locations
3. Establish base classes and interfaces
4. Create configuration management system

### Phase 2: Analytics Expansion
1. Implement comprehensive risk metrics module
2. Implement tail risk metrics (VaR, CVaR, drawdown analysis)
3. Implement factor model analytics (Fama-French, sector analysis)
4. Implement fixed income analytics (duration, convexity, TIPS)

### Phase 3: Optimization Engine
1. Implement base optimizer class
2. Implement traditional optimization (Mean-Variance, Risk Parity)
3. Implement hierarchical optimization (HRP, HERC)
4. Implement Black-Litterman model
5. Implement Kelly criterion optimization
6. Implement shortfall minimization (CVaR/CDaR)
7. Implement constraint system

### Phase 4: Dashboard Development
1. Create main Streamlit application structure
2. Develop reusable components for metrics and visualizations
3. Create individual pages for data ingestion, analytics, optimization, backtesting, and reports
4. Implement state management and caching for dashboard

### Phase 5: Testing and Documentation
1. Implement unit tests for core functionality
2. Implement integration tests for end-to-end workflows
3. Create comprehensive documentation
4. Create example usage notebooks/tutorials

## Integration Points with Existing Code

### Existing ingestion.py
- Migrate to `portfolio_optimizer/data/ingestion.py`
- Enhance with caching and validation layers
- Add error handling and retry logic

### Existing analytics.py
- Split functionality into specialized modules:
  - `risk_metrics.py` (Sharpe, Sortino, Treynor, Information ratios)
  - `tail_risk.py` (VaR, CVaR, Maximum Drawdown)
  - `factor_models.py` (Factor model regression)
  - Keep general utilities in `analytics/__init__.py` or create a `general.py` module

### Configuration
- Use existing pyproject.toml for package metadata
- Add YAML configuration files for runtime parameters
- Maintain backward compatibility with existing dependencies

## Technology Stack
- Core: Python 3.14+
- Data: pandas, numpy, yfinance, fredapi, financetoolkit
- Analytics: scipy, statsmodels
- Optimization: riskfolio-lib, cvxpy
- Dashboard: streamlit, plotly/matplotlib
- Testing: pytest
- Configuration: pyyaml, pydantic (optional)
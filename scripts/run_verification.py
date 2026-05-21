"""Self-contained test runner script to verify all math and solver engines."""

import os
import sys

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.unit.test_analytics import test_risk_metrics, test_tail_risk, test_fixed_income, test_svf_smoothing, test_annuity_payoff
from tests.unit.test_optimization import test_optimizers

def run():
    print("Executing quantitative analytics tests...")
    test_risk_metrics()
    print("[OK] test_risk_metrics passed")
    test_tail_risk()
    print("[OK] test_tail_risk passed")
    test_fixed_income()
    print("[OK] test_fixed_income passed")
    test_svf_smoothing()
    print("[OK] test_svf_smoothing passed")
    test_annuity_payoff()
    print("[OK] test_annuity_payoff passed")
    
    print("\nExecuting optimization engine tests...")
    test_optimizers()
    print("[OK] test_optimizers passed")
    
    print("\n==============================")
    print("ALL VERIFICATION TESTS PASSED!")
    print("==============================")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}", file=sys.stderr)
        sys.exit(1)

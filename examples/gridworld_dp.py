"""Run python -m examples.gridworld_dp; no Gymnasium dependency is needed."""

import argparse

import numpy as np

from modular_rl.dp import policy_evaluation, policy_iteration, uniform_policy, value_iteration
from modular_rl.environment import GRIDWORLD_ACTIONS, sutton_gridworld


def solve_gridworld(theta=1e-10, max_iterations=10_000, max_policy_iterations=100):
    """Use the same known kernel for uniform evaluation and both control algorithms."""
    mdp = sutton_gridworld()
    return {
        "evaluation": policy_evaluation(mdp, uniform_policy(mdp), gamma=1, theta=theta,
                                         max_iterations=max_iterations),
        "policy_iteration": policy_iteration(mdp, gamma=1, theta=theta,
                                              max_iterations=max_policy_iterations,
                                              max_evaluation_iterations=max_iterations),
        "value_iteration": value_iteration(mdp, gamma=1, theta=theta, max_iterations=max_iterations),
    }


def print_actions(policy):
    """Display all positive-probability optimal actions; terminal actions are placeholders."""
    symbols = {name: name[0].upper() for name in GRIDWORLD_ACTIONS}
    for row in range(4):
        cells = []
        for column in range(4):
            state = 4 * row + column
            actions = "T" if state in (0, 15) else "/".join(
                symbols[GRIDWORLD_ACTIONS[action]] for action in np.flatnonzero(policy[state])
            )
            cells.append(f"{actions:12}")
        print(" ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Sutton 4x4 Gridworld: synchronous tabular DP")
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--max-policy-iterations", type=int, default=100)
    args = parser.parse_args()
    results = solve_gridworld(args.theta, args.max_iterations, args.max_policy_iterations)
    references = {
        "evaluation": np.array([0, -14, -20, -22, -14, -18, -20, -20,
                                -20, -20, -18, -14, -22, -20, -14, 0]),
        "optimal": np.array([0, -1, -2, -3, -1, -2, -3, -2,
                             -2, -3, -2, -1, -3, -2, -1, 0]),
    }
    print("Sutton 4x4 Gridworld: gamma=1, synchronous sweeps, terminals=(0, 15)")
    with np.printoptions(precision=6, suppress=True):
        for name, result in results.items():
            print(f"{name}: converged={result.converged} iterations={result.iterations} "
                  f"evaluation_sweeps={result.evaluation_iterations} delta={result.delta:.3e}")
            print(result.values.reshape(4, 4))
            reference = references["evaluation" if name == "evaluation" else "optimal"]
            print(f"max_reference_error={np.max(np.abs(result.values - reference)):.3e}")
            if name != "evaluation":
                label = "optimal actions" if result.converged else "policy actions (not converged)"
                print(f"{name} {label} (U/D/L/R; T=terminal):")
                print_actions(result.policy)
    difference = np.max(np.abs(results["policy_iteration"].values - results["value_iteration"].values))
    print(f"policy_iteration_vs_value_iteration_max_error={difference:.3e}")


if __name__ == "__main__":
    main()

"""
Profitable Betting Strategy Module

Based on EDA findings:
- 1-2 odds: +2% edge
- 5-20 odds: +1.3% edge
- Market inefficiencies can be exploited
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def calculate_expected_value(
    predicted_prob: np.ndarray,
    odds: np.ndarray
) -> np.ndarray:
    """
    Calculate Expected Value (EV) of betting

    EV = (predicted_prob × odds) - 1

    Args:
        predicted_prob: Predicted win probability
        odds: Win odds (e.g., 2.5 means 2.5x return)

    Returns:
        Expected value for each bet
    """
    return (predicted_prob * odds) - 1


def kelly_criterion(
    predicted_prob: np.ndarray,
    odds: np.ndarray,
    fractional: float = 0.25
) -> np.ndarray:
    """
    Calculate Kelly Criterion betting fraction

    Kelly = (odds × prob - 1) / (odds - 1)

    Args:
        predicted_prob: Predicted win probability
        odds: Win odds
        fractional: Fraction of Kelly to use (0.25 = quarter Kelly, safer)

    Returns:
        Optimal betting fraction of bankroll
    """
    kelly = np.zeros_like(predicted_prob)
    mask = odds > 1
    kelly[mask] = (odds[mask] * predicted_prob[mask] - 1) / (odds[mask] - 1)
    kelly = np.clip(kelly, 0, 1)  # Don't bet negative or > 100%
    return kelly * fractional


def simulate_betting(
    y_true: np.ndarray,
    predicted_prob: np.ndarray,
    odds: np.ndarray,
    strategy: str = 'ev',
    ev_threshold: float = 0.0,
    prob_threshold: float = 0.5,
    initial_bankroll: float = 100000.0,
    bet_size_method: str = 'fixed',
    fixed_bet_size: float = 1000.0,
    kelly_fraction: float = 0.25,
) -> Dict[str, Any]:
    """
    Simulate betting strategy on historical data

    Args:
        y_true: Actual outcomes (1 = win, 0 = lose)
        predicted_prob: Predicted win probabilities
        odds: Win odds for each race
        strategy: 'ev' (positive EV only) or 'threshold' (prob > threshold)
        ev_threshold: Minimum EV to place bet (default 0.0)
        prob_threshold: Minimum probability to place bet (for threshold strategy)
        initial_bankroll: Starting bankroll
        bet_size_method: 'fixed' or 'kelly'
        fixed_bet_size: Fixed bet amount (if method='fixed')
        kelly_fraction: Kelly fraction (if method='kelly')

    Returns:
        Dictionary with simulation results
    """
    n_races = len(y_true)
    bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]

    total_bets = 0
    total_wins = 0
    total_wagered = 0
    total_returned = 0

    bet_details = []

    for i in range(n_races):
        prob = predicted_prob[i]
        odd = odds[i]
        actual = y_true[i]

        # Calculate EV
        ev = calculate_expected_value(np.array([prob]), np.array([odd]))[0]

        # Decide whether to bet
        should_bet = False
        if strategy == 'ev':
            should_bet = ev > ev_threshold
        elif strategy == 'threshold':
            should_bet = prob > prob_threshold

        if not should_bet:
            bankroll_history.append(bankroll)
            continue

        # Determine bet size
        if bet_size_method == 'fixed':
            bet_amount = min(fixed_bet_size, bankroll)
        elif bet_size_method == 'kelly':
            kelly_frac = kelly_criterion(
                np.array([prob]),
                np.array([odd]),
                fractional=kelly_fraction
            )[0]
            bet_amount = bankroll * kelly_frac
            bet_amount = max(100, min(bet_amount, bankroll))  # Min 100, max bankroll
        else:
            raise ValueError(f"Unknown bet_size_method: {bet_size_method}")

        if bet_amount < 100 or bankroll < 100:
            # Bankrupt or bet too small
            bankroll_history.append(bankroll)
            continue

        # Place bet
        total_bets += 1
        total_wagered += bet_amount

        if actual == 1:
            # Win
            total_wins += 1
            profit = bet_amount * (odd - 1)
            total_returned += bet_amount * odd
            bankroll += profit
        else:
            # Lose
            bankroll -= bet_amount
            total_returned += 0

        bankroll_history.append(bankroll)

        bet_details.append({
            'race_idx': i,
            'predicted_prob': prob,
            'odds': odd,
            'ev': ev,
            'bet_amount': bet_amount,
            'actual_outcome': actual,
            'profit': bankroll - bankroll_history[-2],
            'bankroll': bankroll,
        })

    # Calculate metrics
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    roi = ((total_returned - total_wagered) / total_wagered * 100) if total_wagered > 0 else 0
    total_profit = bankroll - initial_bankroll
    profit_pct = (total_profit / initial_bankroll * 100)

    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'total_profit': total_profit,
        'profit_pct': profit_pct,
        'total_bets': total_bets,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'total_returned': total_returned,
        'roi': roi,
        'bankroll_history': bankroll_history,
        'bet_details': bet_details,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    predicted_prob: np.ndarray,
    odds: np.ndarray,
    strategy: str = 'ev',
    threshold_range: Tuple[float, float] = (-0.1, 0.2),
    num_steps: int = 50,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal threshold that maximizes ROI

    Args:
        y_true: Actual outcomes
        predicted_prob: Predicted probabilities
        odds: Win odds
        strategy: 'ev' or 'threshold'
        threshold_range: (min, max) threshold to search
        num_steps: Number of thresholds to try
        **kwargs: Additional arguments for simulate_betting

    Returns:
        (optimal_threshold, best_result)
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    best_roi = -float('inf')
    best_threshold = 0.0
    best_result = None

    for threshold in thresholds:
        if strategy == 'ev':
            result = simulate_betting(
                y_true, predicted_prob, odds,
                strategy='ev',
                ev_threshold=threshold,
                **kwargs
            )
        else:
            result = simulate_betting(
                y_true, predicted_prob, odds,
                strategy='threshold',
                prob_threshold=threshold,
                **kwargs
            )

        if result['total_bets'] >= 10:  # Minimum bets required
            if result['roi'] > best_roi:
                best_roi = result['roi']
                best_threshold = threshold
                best_result = result

    return best_threshold, best_result


def analyze_by_odds_range(
    y_true: np.ndarray,
    predicted_prob: np.ndarray,
    odds: np.ndarray,
    odds_ranges: list = None
) -> pd.DataFrame:
    """
    Analyze profitability by odds ranges

    Args:
        y_true: Actual outcomes
        predicted_prob: Predicted probabilities
        odds: Win odds
        odds_ranges: List of (min, max) tuples for odds ranges

    Returns:
        DataFrame with analysis by odds range
    """
    if odds_ranges is None:
        odds_ranges = [
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 5.0),
            (5.0, 10.0),
            (10.0, 20.0),
            (20.0, 50.0),
            (50.0, 1000.0),
        ]

    results = []

    for min_odds, max_odds in odds_ranges:
        mask = (odds >= min_odds) & (odds < max_odds)
        if mask.sum() == 0:
            continue

        odds_subset = odds[mask]
        prob_subset = predicted_prob[mask]
        y_subset = y_true[mask]

        # Calculate metrics
        n_bets = len(y_subset)
        n_wins = y_subset.sum()
        win_rate = n_wins / n_bets if n_bets > 0 else 0

        # Simulate betting (fixed 1000 per bet)
        total_wagered = n_bets * 1000

        # Convert to numpy arrays for indexing
        odds_arr = odds_subset.values if hasattr(odds_subset, 'values') else odds_subset
        y_arr = y_subset.values if hasattr(y_subset, 'values') else y_subset

        total_returned = sum(
            1000 * odds_arr[i] if y_arr[i] == 1 else 0
            for i in range(n_bets)
        )
        roi = ((total_returned - total_wagered) / total_wagered * 100) if total_wagered > 0 else 0

        # Expected value
        ev = calculate_expected_value(prob_subset, odds_subset)
        avg_ev = ev.mean()

        results.append({
            'odds_range': f'{min_odds}-{max_odds}',
            'n_bets': n_bets,
            'n_wins': n_wins,
            'win_rate': win_rate,
            'avg_predicted_prob': prob_subset.mean(),
            'avg_odds': odds_subset.mean(),
            'avg_ev': avg_ev,
            'roi': roi,
            'total_wagered': total_wagered,
            'total_returned': total_returned,
            'profit': total_returned - total_wagered,
        })

    return pd.DataFrame(results)
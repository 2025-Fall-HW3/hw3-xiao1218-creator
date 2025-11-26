"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets_list = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """
# Calculate the number of included assets
        num_assets = len(assets)
        
        # Calculate the weight for each included asset
        if num_assets > 0:
            weight = 1.0 / num_assets
            
            # Assign the calculated weight to the included assets across all dates
            self.portfolio_weights[assets] = weight

        """
        TODO: Complete Task 1 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets_list = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets_list]
            .mul(self.portfolio_weights[assets_list])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        assets_list = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """
          # 1. Calculate the rolling standard deviation (volatility) of returns.
        # This serves as the proxy for the standalone risk (sigma) of each asset
        # over the specified lookback window. We set min_periods to ensure
        # the calculation starts only after the lookback period is met.
        rolling_std = df_returns[assets].rolling(
            window=self.lookback, min_periods=self.lookback
        ).std()

        # 2. Calculate the Inverse Volatility: w_i is proportional to 1 / sigma_i.
        # Assets with higher historical volatility will receive lower weights.
        inv_vol_weights = 1.0 / rolling_std

        # 3. Normalize the Inverse Volatility Weights.
        # We must ensure that the weights sum to 1 across the assets for each day.
        # Sum across the assets (axis=1) for the denominator:
        sum_inv_vol = inv_vol_weights.sum(axis=1)
        
        # Divide each asset's inverse volatility by the total inverse volatility sum:
        normalized_weights = inv_vol_weights.div(sum_inv_vol, axis=0)

        # 4. Assign the calculated weights to the portfolio_weights DataFrame.
        self.portfolio_weights[assets] = normalized_weights
        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets_list = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets_list]
            .mul(self.portfolio_weights[assets_list])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets_list = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets_list].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets_list] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """
                # Initialize Decision w
                # Set lower bound to 0 for long-only constraint (w_i >= 0)
                # The upper bound is already set to 1 by the sample code, which is fine
                w = model.addMVar(n, name="w", lb=0, ub=1)

                # Set Objective: Maximize (w^T * mu - gamma * w^T * Sigma * w)
                objective_expr = w @ mu - gamma * (w @ Sigma @ w)
                model.setObjective(objective_expr, gp.GRB.MAXIMIZE)

                # Add Constraint: Sum of weights equals 1 (Budget constraint)
                model.addConstr(w.sum() == 1, name="budget")
                """
                TODO: Complete Task 3 Above
                """

                model.optimize()

                if model.status in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
                    solution = [model.getVarByName(f"w[{i}]").X for i in range(n)]
                else:
                    solution = [0] * n

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets_list = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets_list]
            .mul(self.portfolio_weights[assets_list])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)

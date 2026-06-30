"""
Feature Engineering Module
--------------------------
Creates customer-level features for churn prediction.
"""

import pandas as pd
import numpy as np


def create_customer_features(df):
    """
    Create customer-level features from transaction data.
    """

    # Reference date
    snapshot_date = df["InvoiceDate"].max()

    # ----------------------------
    # RFM Features
    # ----------------------------

    recency = (
        snapshot_date -
        df.groupby("Customer ID")["InvoiceDate"].max()
    ).dt.days

    frequency = (
        df.groupby("Customer ID")["Invoice"]
        .nunique()
    )

    monetary = (
        df.groupby("Customer ID")["TotalAmount"]
        .sum()
    )

    customer_df = pd.concat(
        [recency, frequency, monetary],
        axis=1
    )

    customer_df.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    # ----------------------------
    # Average Order Value
    # ----------------------------

    customer_df["AverageOrderValue"] = (
        df.groupby("Customer ID")["TotalAmount"]
        .mean()
    )

    # ----------------------------
    # Total Quantity
    # ----------------------------

    customer_df["TotalQuantity"] = (
        df.groupby("Customer ID")["Quantity"]
        .sum()
    )

    # ----------------------------
    # Unique Products
    # ----------------------------

    customer_df["UniqueProducts"] = (
        df.groupby("Customer ID")["StockCode"]
        .nunique()
    )

    # ----------------------------
    # Customer Lifetime
    # ----------------------------

    first_purchase = (
        df.groupby("Customer ID")["InvoiceDate"]
        .min()
    )

    last_purchase = (
        df.groupby("Customer ID")["InvoiceDate"]
        .max()
    )

    customer_df["CustomerLifetime"] = (
        last_purchase -
        first_purchase
    ).dt.days

    # ----------------------------
    # Average Purchase Gap
    # ----------------------------

    purchase_gap = (
        df.sort_values("InvoiceDate")
        .groupby("Customer ID")["InvoiceDate"]
        .diff()
        .dt.days
    )

    avg_gap = purchase_gap.groupby(df["Customer ID"]).mean()

    customer_df["AveragePurchaseGap"] = avg_gap

    customer_df["AveragePurchaseGap"] = (
        customer_df["AveragePurchaseGap"]
        .fillna(0)
    )

    # ----------------------------
    # Churn Label
    # ----------------------------

    customer_df["Churn"] = np.where(
        customer_df["Recency"] > 90,
        1,
        0
    )

    customer_df.reset_index(inplace=True)

    return customer_df


def save_features(customer_df, output_path):
    """
    Save customer feature dataset.
    """

    customer_df.to_csv(
        output_path,
        index=False
    )

    print(f"Customer features saved to {output_path}")
import streamlit as st
from app.prediction import predict_customer

# Page config
st.set_page_config(page_title="Customer Retention System", layout="centered")

# Title
st.title("📊 Intelligent Customer Retention System")
st.write("Predict whether a customer will churn or stay active.")

st.markdown("---")

# Input form
st.subheader("Enter Customer Details")

recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (number of purchases)", min_value=0)
monetary = st.number_input("Monetary (total spending)", min_value=0.0)

avg_order = st.number_input("Average Order Value", min_value=0.0)
total_qty = st.number_input("Total Quantity Purchased", min_value=0)
unique_products = st.number_input("Unique Products Bought", min_value=0)

lifetime = st.number_input("Customer Lifetime (days)", min_value=0)
gap = st.number_input("Average Purchase Gap (days)", min_value=0)

# Predict button
if st.button("🔮 Predict Churn"):

    customer_data = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AverageOrderValue": avg_order,
        "TotalQuantity": total_qty,
        "UniqueProducts": unique_products,
        "CustomerLifetime": lifetime,
        "AveragePurchaseGap": gap
    }

    prediction, probability = predict_customer(customer_data)

    st.markdown("---")

    st.subheader("Prediction Result")

    st.write(f"Churn Probability: **{probability:.2f}**")

    # Risk level logic
    if probability < 0.3:
        st.success("🟢 Low Risk Customer (Likely to stay)")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer (Likely to churn)")

    st.markdown("---")

    st.subheader("Business Recommendation")

    if probability < 0.3:
        st.write("✔ Continue engagement with regular marketing.")
    elif probability < 0.7:
        st.write("📧 Send personalized offers and emails.")
    else:
        st.write("💰 Offer discount / loyalty program / retention call.")
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Load the expected feature names
expected_features = joblib.load("expected_features.pkl")

def predict_fraud(transaction_data):
    # Convert dictionary to DataFrame
    transaction_df = pd.DataFrame([transaction_data])

    # Standardize column names (strip spaces & ensure case consistency)
    transaction_df.columns = transaction_df.columns.str.strip()

    # Reindex to match model's expected features
    transaction_df = transaction_df.reindex(columns=expected_features, fill_value=0)

    # Convert all values to numeric to avoid dtype errors
    transaction_df = transaction_df.apply(pd.to_numeric, errors="coerce")

    #print("Final Processed Data Features:", transaction_df.columns.tolist())  # Debugging

    # Predict fraud probability
    fraud_probability = model.predict_proba(transaction_df)[0][1]
    prediction = "Fraudulent Transaction" if fraud_probability > 0.5 else "Legitimate Transaction"

    return prediction, fraud_probability

if __name__ == "__main__":
    # ************** FRAUD TRANSACTION *******************
    new_transaction = {
        "CUSTOMER_ID": 99999,
        "TERMINAL_ID": 88888,
        "TX_AMOUNT": 5000.0,
        "TX_TIME_SECONDS": 1000,
        "TX_TIME_DAYS": 50,
        "TX_FRAUD_SCENARIO": 1,
        "TRANSACTION_ID": 123456,  # Ensure this field is included
        "TX_DAY": 6,  # Rename properly
        "TX_HOUR": 23  # Rename properly
    }
    # ************** LEGIT TRANSACTION *****************
    # new_transaction = {
    #     "CUSTOMER_ID": 12345,
    #     "TERMINAL_ID": 67890,
    #     "TX_AMOUNT": 50.0,  # Small transaction amount
    #     "TX_TIME_SECONDS": 36000,  # Midday transaction
    #     "TX_TIME_DAYS": 30,
    #     "TX_FRAUD_SCENARIO": 0,  # Legit transactions usually have no fraud scenario
    #     "TX_HOURS": 12,  # A common transaction hour
    #     "TX_DAY_OF_WEEK": 3,  # Mid-week transaction
    #     "HIGH_AMOUNT": 0,  # Not flagged as a high amount
    #     "TERMINAL_FRAUD_HISTORY": 0,  # No history of fraud at terminal
    #     "CUSTOMER_AVG_SPEND": 45.0,  # Close to the transaction amount
    #     "UNUSUAL_SPEND": 0  # No unusual spending behavior
    # }

    result, prob = predict_fraud(new_transaction)
    print(f"Prediction: {result} ")

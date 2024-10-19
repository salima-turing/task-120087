import pandas as pd
from fairlearn.metrics import demographic_parity_difference

# Dummy data for payment transactions
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M'],
    'Age': [25, 30, 28, 35, 40, 27, 32, 45, 38, 24],
    'Amount': [100, 50, 75, 120, 80, 90, 60, 110, 40, 70],
    'Approved': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Define protected attributes and outcome variable
protected_attributes = ['Gender']
outcome_variable = 'Approved'

# Calculate disparate impact difference
dip = demographic_parity_difference(df[outcome_variable], df[protected_attributes])

print("Disparate Impact Difference:", dip)

# Interpretation:
# A positive DIP indicates that the outcome variable is less favorable for certain protected groups compared to others.
# In this example, a DIP of 0.2 suggests that females are less likely to be approved for payments compared to males.

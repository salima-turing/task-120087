import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

class TestPaymentFairness(unittest.TestCase):

	def setUp(self):
		# Dummy data for testing
		data = {
			'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
			'transaction_amount': [10, 20, 30, 15, 25, 35, 40, 28, 12, 32],
			'age': [25, 30, 35, 40, 45, 50, 28, 32, 37, 42],
			'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'M'],
			'payment_outcome': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
		}
		self.df = pd.DataFrame(data)

		self.protected_attribute = 'gender'
		self.target_attribute = 'payment_outcome'

	def test_demographic_parity(self):
		X = self.df.drop(columns=[self.protected_attribute, self.target_attribute])
		y = self.df[self.target_attribute]
		sensitive_features = self.df[self.protected_attribute]

		model = RandomForestClassifier(random_state=0)
		model.fit(X, y)

		dpd = demographic_parity_difference(y_true=y, y_pred=model.predict(X), sensitive_features=sensitive_features)
		self.assertLess(abs(dpd), 0.1, "Demographic parity difference is above the acceptable threshold")

if __name__ == '__main__':
	unittest.main()

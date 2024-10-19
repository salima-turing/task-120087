import unittest
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dummy data for testing
payment_data = pd.DataFrame({
    'amount': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'user_age': [25, 35, 45, 55, 28, 32, 40, 50, 60, 24],
    'is_female': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})


class TestFairnessInPaymentSystem(unittest.TestCase):

    def setUp(self):
        self.data = payment_data.copy()
        self.X = self.data.drop(columns=['label'])
        self.y = self.data['label']
        self.model = RandomForestClassifier(random_state=0).fit(self.X, self.y)
        self.explainer = LimeTabularExplainer(self.X.values, feature_names=self.X.columns,
                                              class_names=['approved', 'declined'])

    def test_fairness_metric(self, max_diff=0.1):
        """
        Test fairness using LIME explanations and demographic parity as the fairness metric
        """

        def compute_demographic_parity(explanations, sensitive_attribute='is_female'):
            parity_values = []
            for i in range(len(explanations)):
                local_pred = explanations[i].local_pred[1]
                if self.data.iloc[i][sensitive_attribute] == 1:
                    parity_values.append(local_pred)

            parity_rate = sum(parity_values) / len(parity_values)
            return parity_rate

        explanations = [self.explainer.explain_instance(x.values, self.model.predict_proba, num_features=3) for x in
                        self.X]

        parity_rate = compute_demographic_parity(explanations)

        self.assertAlmostEqual(parity_rate, 0.5, delta=max_diff, msg="Demographic parity is not fair")


if __name__ == '__main__':
    unittest.main()

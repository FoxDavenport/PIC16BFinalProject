from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.special import inv_boxcox
import statsmodels.api as sm
import numpy as np

class ModelEvaluator:
    """
    A class to evaluate machine learning models.

    Attributes:
    - test_data (pd.DataFrame): The test dataset.
    - top_10_features (list): List of top 10 features to be used in evaluation.
    - best_lambda_out (float): Best lambda value for Box-Cox transformation.
    - model_boxcox_out (object): Trained model using Box-Cox transformed data.
    - best_lambda_out_red (float): Best lambda value for Box-Cox transformation (reduced features).
    - model_boxcox_out_red (object): Trained model using Box-Cox transformed data (reduced features).
    - model_boxcox (object): Trained model without Box-Cox transformation.
    - full_algorithm (object): Trained model with full algorithm.
    """

    def __init__(self, test_data, top_10_features, best_lambda_out, model_boxcox_out,
                 best_lambda_out_red, model_boxcox_out_red, model_boxcox, full_algorithm):
        """
        Initialize the ModelEvaluator.

        Parameters:
        - test_data (pd.DataFrame): The test dataset.
        - top_10_features (list): List of top 10 features to be used in evaluation.
        - best_lambda_out (float): Best lambda value for Box-Cox transformation.
        - model_boxcox_out (object): Trained model using Box-Cox transformed data.
        - best_lambda_out_red (float): Best lambda value for Box-Cox transformation (reduced features).
        - model_boxcox_out_red (object): Trained model using Box-Cox transformed data (reduced features).
        - model_boxcox (object): Trained model without Box-Cox transformation.
        - full_algorithm (object): Trained model with full algorithm.
        """
        self.test_data = test_data
        self.top_10_features = top_10_features
        self.best_lambda_out = best_lambda_out
        self.model_boxcox_out = model_boxcox_out
        self.best_lambda_out_red = best_lambda_out_red
        self.model_boxcox_out_red = model_boxcox_out_red
        self.model_boxcox = model_boxcox
        self.full_algorithm = full_algorithm

    def evaluate_model(self, features, model, best_lambda=None):
        """
        Evaluate the specified model using given features.

        Parameters:
        - features (list): List of features to be used in evaluation.
        - model (object): Trained machine learning model.
        - best_lambda (float): Best lambda value for Box-Cox transformation (optional).

        Returns:
        - None
        """
        # Make predictions
        test_data_algo = self.test_data[features]
        test_data_algo = sm.add_constant(test_data_algo)
        y_pred = model.predict(test_data_algo)

        # Inverse Box-Cox transformation on the predictions
        if best_lambda:
            y_pred = inv_boxcox(y_pred, best_lambda)

        # Actual SalePrice values
        y_test = self.test_data['SalePrice']

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Print the metrics
        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Absolute Percentage Error: {mape:.2f}%')


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

class DiagnosticPlots:
    """
    A class to create diagnostic plots for a fitted OLS model.

    Attributes:
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model.
    leverage : ndarray
        The leverage (hat) values for each observation.
    standardized_residuals : ndarray
        The internally studentized residuals.

    Methods:
    -------
    plot_residuals_vs_fitted():
        Plots residuals vs fitted values.
    plot_qq():
        Plots the Normal Q-Q plot of standardized residuals.
    plot_scale_location():
        Plots the scale-location plot (sqrt(|residuals|) vs fitted values).
    plot_residuals_vs_leverage():
        Plots the residuals vs leverage plot.
    """

    def __init__(self, model):
        """
        Constructs all the necessary attributes for the DiagnosticPlots object.

        Parameters:
        ----------
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            The fitted OLS model.
        """
        self.model = model
        self.leverage = OLSInfluence(model).influence
        self.standardized_residuals = OLSInfluence(model).resid_studentized_internal

    def plot_residuals_vs_fitted(self):
        """
        Plots residuals vs fitted values.
        """
        plt.figure(figsize=(8, 6))
        sns.residplot(x=self.model.fittedvalues, y=self.model.resid, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
        plt.show()

    def plot_qq(self):
        """
        Plots the Normal Q-Q plot of standardized residuals.
        """
        plt.figure(figsize=(6, 6))
        sm.qqplot(self.standardized_residuals, line='45')
        plt.title('Normal Q-Q Plot')
        plt.show()

    def plot_scale_location(self):
        """
        Plots the scale-location plot (sqrt(|residuals|) vs fitted values).
        """
        plt.figure(figsize=(8, 6))
        sns.residplot(x=self.model.fittedvalues, y=np.sqrt(np.abs(self.model.resid)), lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.xlabel('Fitted values')
        plt.ylabel('sqrt(|Residuals|)')
        plt.title('Scale-Location Plot')
        plt.show()

    def plot_residuals_vs_leverage(self):
        """
        Plots the residuals vs leverage plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.leverage, self.standardized_residuals, alpha=0.5)
        sns.regplot(x=self.leverage, y=self.standardized_residuals, scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residuals')
        plt.title('Residuals vs Leverage')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=-2, color='g', linestyle='--')
        plt.axhline(y=2, color='g', linestyle='--')
        plt.show()
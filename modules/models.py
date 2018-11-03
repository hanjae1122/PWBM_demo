from abc import ABCMeta, abstractmethod

import numpy as np
import patsy
import pandas as pd

from scipy.interpolate import interp1d

import statsmodels.api as sm
import statsmodels.duration.hazard_regression as hzrd_reg

from lifelines import CoxTimeVaryingFitter
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.model_selection import StratifiedKFold, GroupKFold


class Supervised(metaclass=ABCMeta):
    # formula is patsy-interpretable formula
    # data is always in individual loan format
    def __init__(self, train, test, formula):
        self._train, self._test = train, test
        self._formula = formula

    @property
    def formula(self):
        return self._formula

    @property
    def train(self):
        return self._train

    # for indv, need indv format
    # otherwise, delete to save mem    
    @train.deleter
    def train(self):
        del self._train

    @property
    def test(self):
        return self._test

    # for indv, need indv format
    # otherwise, delete to save mem
    @test.deleter
    def test(self):
        del self._test

    @property
    def fit(self):
        return self._fit

    @abstractmethod
    def fit_model(self, fit_input):
        pass

    @abstractmethod
    def make_pred(self, pred_input, use_train):
        pass

    @abstractmethod
    def summary(self):
        pass


class OLS(Supervised):
    # Ordinary Least Squares
    def __init__(self, train, test, formula):
        super().__init__(train, test, formula)
        self._y_train, self._X_train = patsy.dmatrices(self.formula,
                                                       self.train,
                                                       return_type='dataframe')

    # fit_input: new data to train on; dflts to train set
    def fit_model(self, fit_input=None):
        model_kwargs = {'hasconst': True}
        if fit_input is None:
            # use fit_input for bootstraps
            (model_kwargs['endog'],
             model_kwargs['exog']) = self._y_train, self._X_train
        else:
            (model_kwargs['endog'],
             model_kwargs['exog']) = patsy.dmatrices(self.formula,
                                                     fit_input,
                                                     return_type='dataframe')
        self._fit = sm.OLS(**model_kwargs).fit()
        return

    # pred_input: new data to make preds on; dflts to test set
    def make_pred(self, pred_input=None, use_train=False):
        pred_kwargs = {}
        if pred_input is None:
            if use_train:
                pred_kwargs['exog'] = self._X_train
            else:
                patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                                'data': self.test,
                                'return_type': 'dataframe'}
                pred_kwargs['exog'] = patsy.dmatrix(**patsy_kwargs)

        else:
            patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                            'data': pred_input,
                            'return_type': 'dataframe'}
            pred_kwargs['exog'] = patsy.dmatrix(**patsy_kwargs)

        return self._fit.predict(**pred_kwargs).ravel()

    def summary(self):
        print(self._fit.summary())

    @property
    def resids(self):
        return self._fit.resid


class GBC(Supervised):
    #GradientBoostingClassifier
    def __init__(self, train, test, formula):
        # keep train, test for bootstrapping
        super().__init__(train, test, formula)
        self._y_train, self._X_train = patsy.dmatrices(self.formula,
                                                       self.train,
                                                       return_type='dataframe')

    # fit_input: new data to train on; dflts to train set
    def fit_model(self, fit_input=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {'loss': 'deviance',
                            'learning_rate': 0.1,
                            'max_depth': 6,
                            'n_estimators': 60}
        if fit_input is None:
            # use fit_input for bootstraps
            y, X = self._y_train, self._X_train
        else:
            y, X = patsy.dmatrices(self.formula,
                                   fit_input,
                                   return_type='dataframe')
        self._fit = (GradientBoostingClassifier(**model_kwargs)
                     .fit(X, y.values.ravel()))
        return

    # pred_input: new data to make preds on; dflts to test set
    def make_pred(self, pred_input=None, use_train=False):
        pred_kwargs = {}
        if pred_input is None:
            if use_train:
                pred_kwargs['X'] = self._X_train
            else:
                patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                                'data': self.test,
                                'return_type': 'dataframe'}
                pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)
        else:
            patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                            'data': pred_input,
                            'return_type': 'dataframe'}
            pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)

        return self._fit.predict_proba(**pred_kwargs)[:, 1].ravel()

    def summary(self):
        print(self._fit.summary())


class RFR(Supervised):
    # RandomForestRegressor
    def __init__(self, train, test, formula):
        # keep train, test for bootstrapping
        super().__init__(train, test, formula)
        self._y_train, self._X_train = patsy.dmatrices(self.formula,
                                                       self.train,
                                                       return_type='dataframe')

    # fit_input: new data to train on; dflts to train set
    def fit_model(self, fit_input=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {'criterion': 'mse',
                            'max_features': 0.333,
                            'n_estimators': 1000}
        if fit_input is None:
            # use fit_input for bootstraps
            y, X = self._y_train, self._X_train
        else:
            y, X = patsy.dmatrices(self.formula,
                                   fit_input,
                                   return_type='dataframe')
        self._fit = (RandomForestRegressor(**model_kwargs)
                     .fit(X, y.values.ravel()))
        return

    # pred_input: new data to make preds on; dflts to test set
    def make_pred(self, pred_input=None, use_train=False):
        pred_kwargs = {}
        if pred_input is None:
            if use_train:
                pred_kwargs['X'] = self._X_train
            else:
                patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                                'data': self.test,
                                'return_type': 'dataframe'}
                pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)
        else:
            patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                            'data': pred_input,
                            'return_type': 'dataframe'}
            pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)

        return self._fit.predict(**pred_kwargs)

    def summary(self):
        print(self._fit.summary())


class GBR(Supervised):
    # GradientBoostingRegressor    
    def __init__(self, train, test, formula):
        # keep train, test for bootstrapping
        super().__init__(train, test, formula)
        self._y_train, self._X_train = patsy.dmatrices(self.formula,
                                                       self.train,
                                                       return_type='dataframe')

    # fit_input: new data to train on; dflts to train set
    def fit_model(self, fit_input=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {'loss': 'ls',
                            'learning_rate': 0.1,
                            'max_depth': 6,
                            'n_estimators': 60}
        if fit_input is None:
            # use fit_input for bootstraps
            y, X = self._y_train, self._X_train
        else:
            y, X = patsy.dmatrices(self.formula,
                                   fit_input,
                                   return_type='dataframe')
        self._fit = (GradientBoostingRegressor(**model_kwargs)
                     .fit(X, y.values.ravel()))
        return

    # pred_input: new data to make preds on; dflts to test set
    def make_pred(self, pred_input=None, use_train=False):
        pred_kwargs = {}
        if pred_input is None:
            if use_train:
                pred_kwargs['X'] = self._X_train
            else:
                patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                                'data': self.test,
                                'return_type': 'dataframe'}
                pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)
        else:
            patsy_kwargs = {'formula_like': self.formula.split('~')[1],
                            'data': pred_input,
                            'return_type': 'dataframe'}
            pred_kwargs['X'] = patsy.dmatrix(**patsy_kwargs)

        return self._fit.predict(**pred_kwargs)

    def summary(self):
        print(self._fit.summary())


class PHR(Supervised):
    #Proportional Hazards Regression
    def __init__(self, train, test, formula, status_name, id_name):
        # keep train, test for bootstrapping
        super().__init__(train, test, formula)
        self._y_train, self._X_train = patsy.dmatrices(self.formula,
                                                       self.train,
                                                       return_type='dataframe')
        self._status_name = status_name
        self._id_name = id_name

    # fit_input: new data to train on; dflts to train set
    def fit_model(self, fit_input=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {'ties': 'efron'}
        if fit_input is None:
            # use fit_input for bootstraps
            (model_kwargs['endog'],
             model_kwargs['exog']) = self._y_train, self._X_train
            model_kwargs['status'] = self.train[self._status_name]
        else:
            # 'status' must be defined in fit_input
            (model_kwargs['endog'],
             model_kwargs['exog']) = patsy.dmatrices(self.formula,
                                                     fit_input,
                                                     return_type='dataframe')
            model_kwargs['status'] = fit_input[self._status_name]

        # delete intercept
        if 'Intercept' in model_kwargs['exog'].columns:
            del model_kwargs['exog']['Intercept']

        self._fit = hzrd_reg.PHReg(**model_kwargs).fit()
        return

    # pred_input: new data to make preds on; dflts to test set
    def make_pred(self, pred_input=None, use_train=False):
        pred_kwargs = {'pred_type': 'hr'}
        if pred_input is None:
            if use_train:
                ages, pred_kwargs['exog'] = self._y_train, self._X_train
                uniq_ids = self.train[self._id_name]
            else:
                ages, pred_kwargs['exog'] = patsy.dmatrices(self.formula,
                                                            self.test,
                                                            return_type='dataframe')
                uniq_ids = self.test[self._id_name]
        else:
            ages, pred_kwargs['exog'] = patsy.dmatrices(self.formula,
                                                        pred_input,
                                                        return_type='dataframe')
            uniq_ids = pred_input[self._id_name]

        # delete intercept
        if 'Intercept' in pred_kwargs['exog'].columns:
            del pred_kwargs['exog']['Intercept']

        # hz_ratio of each loan
        # fxn for baseline cumulative hazard
        hz_ratios = self._fit.predict(**pred_kwargs).predicted_values
        base_cum_hz_fxn = self._fit.baseline_cumulative_hazard_function[0]
        
        hz_df = pd.DataFrame({'ID': uniq_ids, 'hz_r': hz_ratios})
        age_range = np.arange(1, ages.values.max() + 1)
        base_cum_hz = base_cum_hz_fxn(age_range)
        # duplicate ids by len(age_range)
        dup_ids = np.repeat(uniq_ids, len(age_range))
        dup_cum_hz = np.tile(base_cum_hz, len(uniq_ids))
        dup_age = np.tile(age_range, len(uniq_ids))
        base_cum_hz_df = pd.DataFrame({'ID': dup_ids, 'AGE': dup_age,
                                       'base_cum_hz': dup_cum_hz})
        combined = base_cum_hz_df.merge(hz_df, on='ID')
        combined['cum_death_preds'] = 1 - np.exp(-combined['base_cum_hz'] *
                                                 combined['hz_r'])
        del combined['base_cum_hz'], combined['hz_r']
        return combined

    def summary(self):
        print(self._fit.summary())

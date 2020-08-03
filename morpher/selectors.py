from collections import namedtuple
import skrebate
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from boruta import BorutaPy
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.base import BaseEstimator

class NoSelector(object):
    __name__ = "no selector"

    def __init__(self, k, **kwargs):
        pass

    def fit(self, X, y):
        return False

    def fit_transform(self, X, y):
        return X, y

    def __bool(self):
        return False

    def __repr__(self):
        return self.__name__
    
    def get_indices(self):
        return []

class ReliefF(skrebate.ReliefF):
    """
    Extends the standard of ReliefF with standard MORPHER Toolkit interface
    """
    __name__ = "Relief-F"

    def __init__(self, top=10, **kwargs):
        super().__init__(n_features_to_select=top, **kwargs)

    def get_indices(self):
        return self.top_features_[:self.n_features_to_select]

class MutualInfo(SelectKBest):
    """
    Extends the standard implementation SelectPercentile with mutual info initialization
    """
    __name__ = "Mutual Information"

    def __init__(self, top=10, **kwargs):

        super().__init__(mutual_info_classif, k=top, **kwargs)

    def get_indices(self):
        return list(self.get_support(indices=True))

class FTest(SelectKBest):
    """
    Extends the standard implementation SelectPercentile with FTest initialization
    """
    __name__ = "F-Test"

    def __init__(self, top=10, **kwargs):
        super().__init__(f_classif, k=top, **kwargs)
    
    def get_indices(self):
        return list(self.get_support(indices=True))

class Boruta(BorutaPy):
    """
    Extends the standard implementation BorutaPy with default initialization
    """
    __name__ = "Boruta"

    def __init__(self, top=10, **kwargs):
        ''' Boruta finds the top features unsupervisedly '''
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        self.top_ = top
        super().__init__(rf, n_estimators='auto', **kwargs)
    
    def get_indices(self):
        return [idx for idx, val in enumerate(self.support_) if val][:self.top_]

class ElasticNetSelector(BaseEstimator):
    """
    Implements the ElasticNet selector
    """
    __name__ = "Elastic Net"

    def __init__(self, top=10, cross_validate=False, verbose=False, **kwargs):
        kwargs['normalize'] = kwargs.get('normalize') or True
        kwargs['alpha'] = kwargs.get('alpha') or 0.001
        self.classifier_ = ElasticNet(**kwargs)
        self.top_ = top
        self.cross_validate_ = cross_validate
        self.verbose = verbose

    def fit(self, X, y):
        if self.cross_validate_:
            cv_model = ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1],
                eps=0.001,
                n_alphas=100,
                fit_intercept=True, 
                normalize=True,
                precompute='auto',
                max_iter=2000,
                tol=0.0001,
                cv=5, 
                copy_X=True,
                verbose=0,
                n_jobs=-1,
                positive=False,
                selection='cyclic'
            )

            cv_model.fit(X, y)

            if self.verbose:
                print('Optimal alpha: %.8f'%cv_model.alpha_)
                print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
                print('Number of iterations %d'%cv_model.n_iter_)

            self.classifier_ = ElasticNet(
                l1_ratio=cv_model.l1_ratio_,
                alpha = cv_model.alpha_,
                max_iter=cv_model.n_iter_,
                fit_intercept=True,
                normalize = True
            )

        self.classifier_.fit(X, y)
    
    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def __bool(self):
        return False

    def __repr__(self):
        return self.__name__
    
    def get_indices(self):
        return [idx for idx, val in enumerate(self.classifier_.coef_) if val != 0.0][:self.top_]

_options = {
    "RELIEF_F": ReliefF,
    "MUTUAL_INFO": MutualInfo,
    "F_TEST": FTest,
    "NOSELECTOR": NoSelector,
    "BORUTA" : Boruta,
    "ELASTIC_NET": ElasticNetSelector
}

selector_config = namedtuple("options", _options.keys())(**_options)

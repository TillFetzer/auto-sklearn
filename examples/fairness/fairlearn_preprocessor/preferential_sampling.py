import numpy as np
from autosklearn.pipeline.components.base import AutoSklearnComponent
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
    SPARSE
)
from examples.fairness.fairlearn_preprocessor import FairPreprocessor
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

import random

class  PreferentialSampling(FairPreprocessor, AutoSklearnComponent):
    index_sf = []
    def __init__(self,ranker,n_neighbors=0,**kwargs):
        """This preprocessors samples that the data is fair"""
        self.ranker = ranker
        if ranker == "knn":
            self.n_neighbors = n_neighbors
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    @classmethod
    def utils_fairlearn(cls, index_sf):
        cls.index_sf.append(index_sf)

    def fit(self, X, Y=None):
        self.preprocessor = "prefentialsampling"
        self.fitted_ = True
        if self.ranker == "naivebayes":
            from sklearn.naive_bayes import GaussianNB
            self.ranker = GaussianNB().fit(X, Y)
        if self.ranker == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            self.ranker = KNeighborsClassifier(n_neighbors = self.n_neighbors).fit(X,Y)
        return self
    def transform(self, X, y=None):
        if y is None:
            return X
        X = np.array(X)
        protected = np.array(X[:,PreferentialSampling.index_sf]).flatten()
        
        #y = self.y
        if self.preprocessor is None:
            raise NotImplementedError()
       

        #if type not in set(['uniform', 'preferential']):
        #    raise Exception("type must be either 'uniform' or 'preferential'")

        #if type == 'preferential':
        # TODO the issue there is it predict with the same data then learned, which is the same in the pseudocode of the paper
        # but in an example, so it does know if it makes sense
        probs = self.ranker.predict_proba(X)[:,1]  
        

        weights = np.repeat(None, len(y))

        for subgroup in np.unique(protected):
            for c in np.unique(y):

                Xs = np.sum(protected == subgroup)
                Xc = np.sum(y == c)
                try:
                    Xsc = np.sum((protected == subgroup) & (c == y))
                except:
                    print("error")
                Wsc = (Xs * Xc) / (len(y) * Xsc)
                weights[(protected == subgroup) & (y == c)] = Wsc
            

        expected_size =  dict.fromkeys(np.unique(protected))
        for key in expected_size.keys():
            expected_size[key] = dict.fromkeys(np.unique(y))

        for subgroup in expected_size.keys():
            for value in np.unique(y):
                case_weights = weights[(subgroup == protected) & (value == y)]
                case_size = len(case_weights)
                weight = case_weights[0]
                expected_size[subgroup][value] = round(case_size * weight)

        indices = []

        for subgroup in expected_size.keys():
            for value in np.unique(y):
                current_case = np.arange(len(y))[(protected == subgroup) & (y == value)]
                expected = expected_size[subgroup][value]
                actual = np.sum((protected == subgroup) & (y == value))
                if expected == actual:
                    indices += list(current_case)

                elif expected < actual:
                   
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    if value == 0:
                        indices += list(sorted_current_case[:expected])
                    if value == 1:
                        indices += list(sorted_current_case[-expected:])
                else:
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    p_ind = list(np.repeat(current_case, expected // actual))

                    if expected % actual != 0:
                        if value == 0:
                            p_ind += list(sorted_current_case[-(expected % actual):])
                        if value == 1:
                            p_ind += list(sorted_current_case[:(expected % actual)])
                    indices += p_ind
        #these is a small own, extention,which does not conflict the pseudocode of the original paper
        indices = random.shuffle(indices)
        return np.squeeze(X[indices,:]), np.squeeze(y[indices,:])
        


        #these is in the predict phase 
    
        #return X
    
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "ps",
            "name": "no",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        ranker = CategoricalHyperparameter("ranker", ["naivebayes","knn"],default_value="naivebayes")
        n_neighbors = UniformIntegerHyperparameter("n_neighbors",1,100,default_value=1)
        cs.add_hyperparameters([ranker,n_neighbors])
        knn_cond = EqualsCondition(
        n_neighbors, ranker, "knn"
        )
        cs.add_conditions([knn_cond])
        return cs
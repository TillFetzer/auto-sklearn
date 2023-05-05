from typing import Dict, Optional

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from examples.fairness.fairlearn_preprocessor.abstract_fair_preprocessor import  FairPreprocessor


from autosklearn.pipeline.components.base import (
    BaseEstimator,
    AutoSklearnChoice,
    AutoSklearnComponent,
    ThirdPartyComponents,
    _addons,
    find_components,
)

#rescaling_directory = os.path.split(__file__)[0]
#_rescalers = find_components(
#    __package__, rescaling_directory, AutoSklearnComponent
#)
additional_components = ThirdPartyComponents(AutoSklearnComponent)
_addons["fair_preprocessor"] = additional_components


def add_fair_preprocessor(fair_preprocessor: FairPreprocessor) -> None:
    additional_components.add_component(fair_preprocessor)


class FairChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls: BaseEstimator) -> Dict[str, BaseEstimator]:
        components: Dict[str, BaseEstimator] = OrderedDict()
        #these is benefical
        #components.update(_rescalers)
        components.update(additional_components.components)
        return components
   

    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        default: Optional[str] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError("No rescalers found, please add any rescaling component.")

        #if default is None:
        #    defaults = ["sampling"]
        #    for default_ in defaults:
        ##        if default_ in available_preprocessors:
        #            default = default_
        #            break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[
                name
            ].get_hyperparameter_search_space(
                feat_type=feat_type, dataset_properties=dataset_properties
            )
            #not sure about that 
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs
    def fit(self, X, y):
        return self.choice.fit(X,y)
    def transform(self, X, y = None):
        return self.choice.transform(X,y)
   
    def fit_transform(self, X, y,**fit_params):
        return self.fit(X,y).transform(X,y)

import traceback
import numpy as np
from collections import defaultdict, OrderedDict, namedtuple
from morpher.plots import plot_feat_importances
from lime import lime_tabular, submodular_pick
from sklearn.linear_model import BayesianRidge
import shap


class Base:
    def __init__(self, data, model, target):

        """
        Base interpreter
        """
        self.explainer = None

        """
        Feature explanations, it'll be list of explanations
        """
        self.explanations = []

        """
        Underlying data on which to run the classifier
        """
        self.data = data

        """
        Model for which to run the required explanation
        """
        self.model = model

        """
        Required target
        """
        self.target = target

    def _initialize(self):
        raise NotImplementedError(
            "Please implement the initializer for this method!"
        )

    def plot(self, ax=None):

        """
        Plots a given explanation.
        """
        try:
            plot_feat_importances(self.explanation, ax)
        except Exception:
            pass

    def _append_explanation(self, exp):
        if exp:
            vals = [x[1] for x in exp]
            names = [x[0] for x in exp]
            vals.reverse()
            names.reverse()
            feat_importances = OrderedDict(
                sorted(zip(names, vals), key=lambda x: x[1], reverse=True)
            )
            self.explanations.append(feat_importances)

    @property
    def explanation(self):
        # syntatic sugar for explanation
        return self.explanations[0]


class LimeExplainer(Base):
    def __init__(self, data, model, target, **kwargs):

        super().__init__(data, model, target)

        """
        Initialize explainer
        """
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):

        print("Setting up LIME explainer")
        features = self.data.drop([self.target], axis=1)
        self.explainer = lime_tabular.LimeTabularExplainer(
            features,
            feature_names=kwargs.get("feature_names")
            or list(features.columns),
            class_names=kwargs.get("feature_names")
            or ["Outcome (no)", "Outcome (yes)"],
            mode=kwargs.get("mode") or "classification",
            discretize_continuous=kwargs.get("discretize_continuous") or False,
        )

    def explain(self, **kwargs):

        """
        Explains a given model, optionally performing SubmodularPick .
        """

        index = kwargs.get("index") or None
        # sample size is comprised of 25% of total dataset by default
        sample_size = kwargs.get("sample_size") or round(
            self.data.shape[0] * 0.25
        )
        num_features = kwargs.get("num_features") or 10
        num_exps_desired = kwargs.get("num_exps_desired") or 10
        print_exps = kwargs.get("print_exps") or False

        features = self.data.drop([self.target], axis=1)

        if index is None:
            print("*** Generating explanations using Submodular Pick...")
            print("Sample size chosen: {}".format(round(sample_size)))

            sp_obj = submodular_pick.SubmodularPick(
                self.explainer,
                np.asarray(features),
                self.model.predict_proba,
                sample_size=sample_size,
                num_features=num_features,
                num_exps_desired=num_exps_desired,
            )

            for sp_exp in sp_obj.sp_explanations:
                exp = sp_exp.as_list(label=sp_exp.available_labels()[0])
                self._append_explanation(exp)

            # create averaged contribution for all features across all explanations
            # obtain feature importance in terms of magnitude, discarding signal
            # remove features that appear less than 10% of the time
            explanation = defaultdict(lambda: [])
            for exp in self.explanations:
                for feature in exp:
                    explanation[feature].append(exp[feature])
            avg_explanation = {
                feature: np.mean(np.abs(np.array(explanation[feature])))
                for feature in explanation
                if len(explanation[feature]) > num_exps_desired * 0.25
            }
            avg_explanation = OrderedDict(
                sorted(
                    avg_explanation.items(), key=lambda x: x[1], reverse=True
                )
            )

            for column, value in avg_explanation.items():
                if print_exps:
                    print("{0} = {1}".format(column, value))
            return avg_explanation

        else:
            try:
                print(f"Explaining prediction for case #{index}")
                row_feat = np.asarray(features)[int(index), :].reshape(1, -1)
                y_true, y_pred, y_prob = (
                    self.data[self.target][int(index)],
                    self.model.predict(row_feat),
                    self.model.predict_proba(row_feat),
                )
                print(
                    "Model predicted class {0} with class score {1:.3f}".format(
                        y_pred, y_prob[0, int(y_pred)]
                    )
                )
                print("Actual class is {0}".format(y_true))
                exp = self.explainer.explain_instance(
                    row_feat[0], self.model.predict_proba
                )
                self._append_explanation(
                    exp.as_list(label=exp.available_labels()[0])
                )
            except Exception as e:
                print("Error occurred: {}".format(str(e)))
                print(traceback.format_exc())

        return self.explanation


class FeatContribExplainer(Base):
    def __init__(self, data, model, target, **kwargs):

        super().__init__(data, model, target)

        """
        Initialize explainer
        """
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        # nothing to initialize, model-based
        pass

    def explain(self, **kwargs):

        """
        Displays feature importances of a model if it has feature_importances_
        """

        num_features = kwargs.get("num_features") or 10
        print_exps = kwargs.get("print_exps") or False

        if hasattr(self.model, "feature_importances_"):
            print("*** Obtaining feature importances via classifier:")
            columns = self.data.drop(self.target, axis=1).columns
            exp = sorted(
                zip(columns, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:num_features]
            for column, value in exp:
                if print_exps:
                    print("{0} = {1}".format(column, value))
            self._append_explanation(exp)
        else:
            raise AttributeError(
                "Model does not support feature contribution, please train a different model."
            )

        return self.explanation


class MimicExplainer(Base):
    def __init__(self, data, model, target, **kwargs):

        super().__init__(data, model, target)

        self.mimic = kwargs.get("mimic") or BayesianRidge()

        """
        Initialize explainer
        """
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):

        """
        Displays feature importances of a model if it has feature_importances_
        """

        print("Setting up Mimic explainer")
        features = self.data.drop([self.target], axis=1)
        y_probs = self.model.predict_proba(features)
        self.mimic = self.mimic.fit(features, y_probs[:, 1])

    def explain(self, **kwargs):

        """
        Displays feature importances of the mimic model
        """

        num_features = kwargs.get("num_features") or 10
        print_exps = kwargs.get("print_exps") or False

        if hasattr(self.mimic, "coef_"):
            print("*** Obtaining feature importances via mimic classifier:")
            columns = self.data.drop(self.target, axis=1).columns
            exp = sorted(
                zip(columns, self.mimic.coef_),
                key=lambda x: x[1],
                reverse=True,
            )[:num_features]
            for column, value in exp:
                if print_exps:
                    print("{0} = {1}".format(column, value))
            self._append_explanation(exp)
        else:
            raise AttributeError(
                "Model does not support feature contribution, please train a different model."
            )

        return self.explanation


class ShapExplainer(Base):
    def __init__(self, data, model, target, **kwargs):

        super().__init__(data, model, target)

        """
        Initialize explainer
        """
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):

        features = self.data.drop([self.target], axis=1)
        nsamples = kwargs.get("nsamples") or 100

        if self.model.is_tree_:
            print("Setting up SHAP TreeExplainer")
            self.explainer = shap.TreeExplainer(self.model.clf)
        else:
            print("Setting up SHAP KernelExplainer")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                np.asarray(features),
                nsamples=nsamples,
            )

    def explain(self, **kwargs):

        """
        Explains a given model, using Shap
        """
        test = kwargs.get("test")
        test = test.drop([self.target], axis=1)
        num_features = kwargs.get("num_features") or 10
        print_exps = kwargs.get("print_exps") or False
        columns = self.data.drop(self.target, axis=1).columns

        shap_values = self.explainer.shap_values(np.asarray(test))

        if not self.model.is_tree_:
            shap_values = shap_values[0]

        shap_values = np.abs(shap_values).mean(axis=0).ravel()

        exp = sorted(
            zip(columns, shap_values), key=lambda x: x[1], reverse=True
        )[:num_features]
        for column, value in exp:
            if print_exps:
                print("{0} = {1}".format(column, value))
        self._append_explanation(exp)

        return self.explanation


_options = {
    "LIME": LimeExplainer,
    "MIMIC": MimicExplainer,
    "SHAP": ShapExplainer,
    "FEAT_CONTRIB": FeatContribExplainer,
}

explainer_config = namedtuple("options", _options.keys())(**_options)

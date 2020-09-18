from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    auc,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import MinMaxScaler

from morpher.metrics import (
    get_discrimination_metrics,
    get_net_benefit_metrics,
)

prop_cycle = cycler(
    "color",
    [
        u"#1f77b4",
        u"#ff7f0e",
        u"#2ca02c",
        u"#d62728",
        u"#9467bd",
        u"#8c564b",
        u"#e377c2",
    ],
) + cycler(
    "linestyle",
    [
        (0, ()),
        (0, (1, 5)),
        (0, (1, 1)),
        (0, (5, 5)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
    ],
)


def plot_roc(
    results,
    title="Receiver Operating Curve",
    ax=None,
    figsize=None,
    legend_loc=None
):
    """
    Plots the receiver operating curve of currently loaded results in a new
    window.
    """
    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    ax.plot((0, 1), (0, 1), "k--", label=None)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    plt.rc("axes", prop_cycle=prop_cycle)

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]
        fpr, tpr, thresh = roc_curve(y_true, y_probs)
        # for compatibility issues
        if type(clf_name) == str:
            clf_label = clf_name
        else:        
            clf_label = clf_name().__class__.__name__
            
        ax.plot(
            fpr,
            tpr,
            label="{0} (AUC={1:.2f})".format(
                clf_label, roc_auc_score(y_true, y_probs)
            ),
        )

    ax.legend(loc=legend_loc, fancybox=True, shadow=True)


def plot_prc(
    results,
    title="Precision-Recall Curve",
    ax=None,
    figsize=None,
    legend_loc=None,
):
    """
    Plots the precision recall curve currently loaded results in a new
    window.
    """
    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    plt.rc("axes", prop_cycle=prop_cycle)

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        
        # for compatibility issues
        if type(clf_name) == str:
            clf_label = clf_name
        else:        
            clf_label = clf_name().__class__.__name__
            
        ax.step(
            recall,
            precision,
            label="{0} (AP={1:.2f})".format(
                clf_label, average_precision_score(y_true, y_probs)
            ),
            where="post",
        )

    no_skill_ratio = y_true.sum() / len(y_true)
    ax.plot(
        [0, 1],
        [no_skill_ratio, no_skill_ratio],
        label="No skill",
        color="lightgray",
        linestyle="--",
    )

    ax.legend(loc=legend_loc, fancybox=True, shadow=True)
    ax.autoscale(enable=True)


def plot_cc(
    models,
    train_data,
    test_data,
    target,
    title="Calibration Plot",
    ax=None,
    figsize=None,
    legend_loc=None,
    verbose=False
):
    """
    Plots calibration curve, we need the original train dataset to do this (!)
    """
    if not models:
        raise AttributeError("No models available")

    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    y_train = train_data[target]
    X_train = train_data.drop(target, axis=1)

    y_test = test_data[target]
    X_test = test_data.drop(target, axis=1)

    plt.rc("axes", prop_cycle=prop_cycle)

    for clf_name in models:

        clf = models[clf_name].clf

        calibrated_clf_sig = CalibratedClassifierCV(
            clf, cv="prefit", method="sigmoid"
        )
        calibrated_clf_sig.fit(X_train, y_train)

        calibrated_clf = CalibratedClassifierCV(
            clf, cv="prefit", method="isotonic"
        )
        calibrated_clf.fit(X_train, y_train)

        for clf, name in [
            (clf, clf.__class__.__name__),
            (calibrated_clf, clf.__class__.__name__ + " + isotonic"),
            (calibrated_clf_sig, clf.__class__.__name__ + " + sigmoid"),
        ]:

            y_probs = clf.predict_proba(X_test)
            score = brier_score_loss(
                y_test, y_probs[:, 1], pos_label=y_test.max()
            )
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_probs[:, 1], n_bins=10
            )
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                label="%s (Brier=%1.3f)" % (name, score),
                marker="s",
            )

            score = brier_score_loss(
                y_test, y_probs[:, 1], pos_label=y_test.max()
            )
            if verbose:
                print("*** Brier for %s: %1.3f" % (name, score))

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc=legend_loc, fancybox=True, shadow=True)
    ax.set_title(title)
    
    if verbose:
        print("*** Model calibration performed.\n")


def plot_dc(
    results,
    tr_start=0.01,
    tr_end=0.99,
    tr_step=0.01,
    metric_type="treated",
    ymin=-0.1,
    title="Decision Curve",
    ax=None,
    figsize=None,
    legend_loc=None,
):
    """
    Plots decision curve for treating a patient based on the predicted probability

    Params:

    results               outputs of a given classifier
    tr_start              start of thresholds
    tr_end                end of thresholds
    tr_end                step_size of thresholds
    metric_type           type of metric to be plotted, treated, untreated or adapt
    ymin                  mininum net benefit to be plotted

    """
    if not results:
        raise AttributeError("No results available")

    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    plt.rc("axes", prop_cycle=prop_cycle)
    ax.set_xticks(np.arange(0, 1.25, step=0.25))

    tr_probs = np.arange(tr_start, tr_end + tr_step, tr_step)

    """
    Necessary to decide where to fix ymax
    """
    ymax = 0.0

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]

        net_benefit, net_benefit_treated_all = get_net_benefit_metrics(
            y_true,
            y_probs,
            tr_probs,
            metric_type
        )

        if results[clf_name].get("label"):
            label = results[clf_name]["label"]
        else:
            if type(clf_name) == str:
                label = clf_name
            else:
                label = clf_name().__class__.__name__

        ax.plot(
            tr_probs,
            net_benefit,
            label="{0} (ANBC={1:.2f})".format(
                label, auc(tr_probs, net_benefit)
            ),
        )

        """
        Necessary to decide where to fix ymax
        """
        current_max = np.amax(net_benefit + net_benefit_treated_all)
        if current_max > ymax:
            ymax = current_max

    ax.plot(tr_probs, net_benefit_treated_all, label=metric_type + " (all)")
    ax.axhline(
       y=0.0, color="gray", linestyle="--", label=metric_type + " (none)"
    )
    ax.legend(loc=legend_loc, fancybox=True, shadow=True)

    """
    Define plotting limits
    """
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([0, 1])


def plot_feat_importances(
    feat_importances,
    ax=None,
    friendly_names=None,
    title="Feature Importance",
    figsize=None,
    legend_loc=None,
):
    """
    Plots feat importances based on given explanations

    Params:

    feat_importances      outputs of an explainer, consisting of features and their importance
    ax                    figure axis to draw on

    """
    if not feat_importances:
        raise AttributeError("No feature importances available")

    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    plt.rc("axes", prop_cycle=prop_cycle)

    names, vals = (
        list(feat_importances.keys()),
        list(feat_importances.values()),
    )
    names.reverse()
    vals.reverse()

    if friendly_names:
        names = [
            friendly_names.get(feat_name) or feat_name for feat_name in names
        ]

    colors = [u"#1f77b4" if x > 0 else "#ff7f0e" for x in vals]
    pos = list(range(len(feat_importances)))
    ax.barh(names, vals, align="center", color=colors)
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_title(title)


def plot_weighted_explanation(
    explanations,
    ax=None,
    friendly_names=None,
    top_features=15,
    title="Weighted Feature Importances",
    figsize=None,
    legend_loc=None,
):
    """
    Plots a heatmap based on an array with different feature importances

    Params:

    feat_importances      dict of dicts with outputs of an explainer, where the keys are the explainer method used
    ax                    figure axis to draw on
    friendly_names        if features have a friendly names, use them

    """
    if not explanations:
        raise AttributeError("No explanations available")

    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    plt.rc("axes", prop_cycle=prop_cycle)

    # list of available interpretability methods
    methods = list(explanations.keys())

    # list of all features mentioned across all methods
    features = defaultdict(lambda: [])

    weights = {}

    # first normalize all feature importance values between 0.1 and 1
    for method in methods:
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        importances = np.array(list(explanations[method].values())).reshape(
            -1, 1
        )
        scaler.fit(importances)
        n_importances = list(scaler.fit_transform(importances).ravel())
        explanations[method].update(
            {
                key: value
                for key, value in zip(
                    list(explanations[method].keys()), n_importances
                )
            }
        )

        for feat in explanations[method]:
            features[feat].append(explanations[method][feat])

    # calculate mean while retaining the weights
    for feat in features:
        weights[feat] = len(features[feat])
        features[feat] = float(np.mean(features[feat]))

    pos = list(range(len(features)))

    names, vals = list(features.keys()), list(features.values())
    if friendly_names:
        names = [
            friendly_names.get(feat_name) or feat_name for feat_name in names
        ]

    exps = list(sorted(zip(names, vals), key=lambda x: x[1]))

    # normalize weights
    scaler = MinMaxScaler(feature_range=(0.3, 1))
    scaler.fit(np.array(list(weights.values())).reshape(-1, 1))
    weights = {k: scaler.transform([[v]])[0][0] for k, v in weights.items()}

    color = (31, 119, 180)
    colors = [
        tuple(((c + (255 - c) * (1 - weights[feat])) / 255) for c in color)
        for feat in [exp[0] for exp in exps]
    ]

    ax.barh(
        [exp[0] for exp in exps],
        [exp[1] for exp in exps],
        align="center",
        color=colors,
    )
    ax.set_yticks(pos)
    ax.set_yticklabels([exp[0] for exp in exps])

    # needed to manually set the legend
    color_patch = sorted(list(set(colors)), key=lambda x: x)
    weights = sorted(
        list(set(weights.values())), key=lambda x: x, reverse=True
    )
    label_patch = [
        "Support: {0}".format(
            int(round(scaler.inverse_transform([[w]])[0][0]))
        )
        for w in weights
    ]
    patches = zip(color_patch, label_patch)
    ax.legend(
        handles=[
            mpatches.Patch(color=color, label=label)
            for color, label in patches
        ]
    )
    ax.set_title(title)


def plot_explanation_heatmap(
    explanations,
    ax=None,
    friendly_names=None,
    top_features=15,
    title="Feature Importances by Method",
    figsize=None,
    legend_loc=None,
    valfmt="{x:.2f}",
    cbarlabel="Feature Importance (Outcome=yes)",
):
    """
    Plots a heatmap based on an array with different feature importances

    Params:

    feat_importances      dict of dicts with outputs of an explainer, where the keys are the explainer method used
    ax                    figure axis to draw on
    friendly_names        if features have a friendly names, use them

    """
    if not explanations:
        raise AttributeError("No explanations available")

    if figsize:
        plt.clf()
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    plt.rc("axes", prop_cycle=prop_cycle)

    # list of available interpretability methods
    methods = list(explanations.keys())

    # list of all features mentioned across all methods
    features = []

    # first normalize all feature importance values between 0,1
    for method in methods:
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        importances = np.array(list(explanations[method].values())).reshape(
            -1, 1
        )
        scaler.fit(importances)
        n_importances = list(scaler.fit_transform(importances).ravel())
        explanations[method].update(
            {
                key: value
                for key, value in zip(
                    list(explanations[method].keys()), n_importances
                )
            }
        )

    # now, create a list with all the features needed
    for method in methods:
        for feature in explanations[method]:
            if feature not in features:
                features.append(feature)

    # then, go ahead and np.nan out the same features for the other methods
    for feature in features:
        for method in methods:
            if feature not in explanations[method]:
                explanations[method][feature] = 0.0

    # then, go ahead and np.nan out the same features for the other methods
    feature_means = []
    for feature in features:
        feature_means.append(
            (
                feature,
                np.mean([explanations[method][feature] for method in methods]),
            )
        )
    feature_means = sorted(feature_means, key=lambda x: x[1], reverse=True)
    display_features = [key for key, value in feature_means[:top_features]]

    # build / shape the data array in the order it should appear
    data = np.array(
	[[explanations[method][feature] for method in explanations] for feature in display_features]
    )

    # if we got friendly names, we substitute it here
    if friendly_names:
        display_features = [
            friendly_names.get(feat_name) or feat_name
            for feat_name in display_features
        ]
    
    if methods[0].__class__.__name__ == 'type':
        methods = [m.__name__ for m in methods]

    im, cbar = heatmap(
        data, display_features, methods, ax=ax, cmap="YlGn", cbarlabel=cbarlabel
    )
    
    annotate_heatmap(im, valfmt=valfmt) 
    ax.set_aspect("auto")

    """
        Credit: Copyright 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2018 The Matplotlib development team
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]).tolist())
    ax.set_yticks(np.arange(data.shape[0]).tolist())
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor"
    )

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks((np.arange(data.shape[1] + 1) - 0.5).tolist(), minor=True)
    ax.set_yticks((np.arange(data.shape[0] + 1) - 0.5).tolist(), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

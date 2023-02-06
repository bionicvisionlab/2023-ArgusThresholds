import numpy as np
import scipy.stats as spst
import pulse2percept.implants as p2pi
import matplotlib.pyplot as plt

from .model import predict_fit, predict_cv


__all__ = ['scatter_correlation', 'plot_thresholds', 'plot_tune_results']


def scatter_correlation(xvals, yvals, ax, xticks=[], yticks=[], marker=None,
                        color=None, textloc='upper right'):
    """Scatter plots some data points and fits a regression curve to them"""
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)

    # Ignore NaN:
    isnan = np.isnan(xvals) | np.isnan(yvals)
    xvals = xvals[~isnan]
    yvals = yvals[~isnan]
    assert np.all(xvals.shape == yvals.shape)
    # Scatter plot the data:
    ax.scatter(xvals, yvals, marker=marker, s=50,
               c=color, edgecolors='white', alpha=0.5)

    # Set axis properties:
    if len(xticks) > 0:
        x_range = np.max(xticks) - np.min(xticks)
        xlim = (np.min(xticks) - 0.1 * x_range, np.max(xticks) + 0.1 * x_range)
        ax.set_xticks(xticks)
        ax.set_xlim(*xlim)
    if len(yticks) > 0:
        y_range = np.max(yticks) - np.min(yticks)
        ylim = (np.min(yticks) - 0.1 * y_range, np.max(yticks) + 0.1 * y_range)
        ax.set_yticks(yticks)
        ax.set_ylim(*ylim)

    # Need at least two data points to fit the regression curve:
    if len(xvals) < 2:
        return

    # Fit the regression curve:
    slope, intercept, rval, pval, _ = spst.linregress(xvals, yvals)
    fit = lambda x: slope * x + intercept
    ax.plot([np.min(xvals), np.max(xvals)], [
            fit(np.min(xvals)), fit(np.max(xvals))], 'k--')

    # Annotate with fitting results:
    pvalstr = ("%.2e" % pval) if pval < 0.001 else ("%.03f" % pval)
    if textloc == 'lower right':
        a = ax.axis()
        xt = np.max(xticks) if len(xticks) > 0 else a[1]
        yt = np.min(yticks) if len(yticks) > 0 else a[2]
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f, $p$=%s" % (len(yvals), rval, pvalstr),
                va='bottom', ha='right')
    elif textloc == 'upper left':
        a = ax.axis()
        xt = np.min(xticks) if len(xticks) > 0 else a[0]
        yt = np.max(yticks) if len(yticks) > 0 else a[3]
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f, $p$=%s" % (len(yvals), rval, pvalstr),
                va='top', ha='left')
    elif textloc == 'upper right':
        a = ax.axis()
        xt = np.max(xticks) if len(xticks) > 0 else a[1]
        yt = np.max(yticks) if len(yticks) > 0 else a[3]
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f, $p$=%s" % (len(yvals), rval, pvalstr),
                va='top', ha='right')
    else:
        raise ValueError('Unknown text location "%s"' % textloc)


def plot_thresholds(Xy, subject, date=None, ax=None):
    for col in ['PatientID', 'ElectrodeLabel', 'Thresholds (µA)']:
        if col not in Xy.columns:
            raise ValueError("Xy must have column '%s'" % col)
    if date is None:
        Xy = Xy[Xy['PatientID'] == subject]
    else:
        if 'TestDate (YYYYmmdd)' not in Xy.columns:
            raise ValueError("Xy must have column 'TestDate (YYYYmmdd)'")
        Xy = Xy[(Xy['PatientID'] == subject)
                & (Xy['TestDate (YYYYmmdd)'] == date)]
    implant = p2pi.ArgusII()
    x_center = np.unique([e.x_center for e in implant])
    y_center = np.unique([e.y_center for e in implant])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    for e in implant:
        ax.scatter(e.x_center, e.y_center, marker='o', s=600, linewidth=2,
                   edgecolor='k', facecolor='w')
        ename = '%s%02d' % (e.name[0], int(e.name[1:]))
        row = Xy.loc[Xy['ElectrodeLabel'] == ename, :]
        if not row.empty:
            th = row['Thresholds (µA)'].values[0]
            if np.isnan(th) or int(th) >= 999:
                ax.scatter(e.x_center, e.y_center, marker='x', s=300, c='k')
            else:
                ax.text(e.x_center, e.y_center, str(int(th)), ha='center',
                        va='center')
    ax.set_xlim(-2900, 2800)
    ax.set_ylim(-1700, 1800)
    ax.set_xticks([])
    ax.set_yticks([])
    for c, xc in enumerate(x_center):
        ax.text(xc, y_center[-1] + 350, '%02d' % (c + 1), ha='center',
                va='top')
    for c, yc in enumerate(y_center):
        ax.text(x_center[0] - 330, yc, chr(70 - c), ha='left', va='center')


def plot_tune_results(X, y, model, iinit_params, groups):
    y_pred_fit = predict_fit(X, y, model, iinit_params)
    y_pred_cv = predict_cv(X, y, model, iinit_params, groups)

    idx = (~np.isnan(y_pred_fit)) & (~np.isnan(y_pred_cv))

    fig, axes = plt.subplots(ncols=2, sharex=False,
                             sharey=False, figsize=(15, 6))
    for ax, y_pred, title in zip(axes, [y_pred_fit, y_pred_cv], ['fit', 'cv']):
        scatter_correlation(y[idx], y_pred[idx], ax)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
    fig.tight_layout()
    return fig, axes

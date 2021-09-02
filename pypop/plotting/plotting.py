import copy
import matplotlib as mpl
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from ..utils import metrics

THRESHOLD = 0.05  # Default threshold for small values


SIGCONF_RCPARAMS= {
    # Fig params
    "figure.autolayout": True,          # Makes sure nothing the feature is neat & tight.
    "figure.figsize": (5.5, 2.95),      # Column width: 3.333 in, space between cols: 0.333 in.
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,
    "axes.labelpad": 0.5,
    # Plot params
    "lines.linewidth": 0.8,              # Width of lines
    "lines.markeredgewidth": 0.3,
    "errorbar.capsize": 0.5,
    # Legend params
    "legend.fontsize": 8.5,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # Font params
    "text.usetex": True,                 # use LaTeX to write all text
    "font.family": "serif",              # use serif rather than sans-serif
    "font.serif": "Linux Libertine O",   # use "Linux Libertine" as the standard font
    "font.size": 9,
    "axes.titlesize": 8,          # LaTeX default is 10pt font.
    "axes.labelsize": 8,          # LaTeX default is 10pt font.
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    # PDF settings
    "pgf.texsystem": "xelatex",         # Use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r"\usepackage{amsmath}",
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\usepackage{libertine}',
        r'\setmainfont{Linux Libertine O}',
        r'\setmathfont{Linux Libertine O}',
    ]
}

ICML_RCPARAMS = {
    # Fig params
    "figure.autolayout": True,          # Makes sure nothing the feature is neat & tight.
    "figure.figsize": (3.25, 2.95),     # Page width: 6.75in, space between cols: 0.25 in, column width: 3.25 in
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times",              # use "Times" as the standard font for ICML
    "font.size": 7,
    "axes.titlesize": 7,                # LaTeX default is 10pt font.
    "axes.labelsize": 7,                # LaTeX default is 10pt font.
    "legend.fontsize": 7,               # Make the legend/label fonts a little smaller
    "legend.frameon": True,            # Remove the black frame around the legend
    "patch.linewidth": 0.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "grid.linewidth": 0.3,
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\setmainfont{Times}',
    ]
}

def set_notebook_config():
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
    mpl.rcParams.update(SIGCONF_RCPARAMS)
    np.set_printoptions(edgeitems=10, linewidth=1000)


def set_icml_config():
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
    mpl.rcParams.update(ICML_RCPARAMS)
    np.set_printoptions(edgeitems=10, linewidth=1000)


def make_metric(df, func, col, col_true='adjacency', **kwargs):
    def processed_func(row):
        return func(row[col].flatten(), row[col_true].flatten(), **kwargs)
    return df.apply(processed_func, axis=1)


def make_perf(df, func, prefix, suf_col_dict, **kwargs):
    col_list = list()
    for suf, col in suf_col_dict.items():
        name = '_'.join([prefix, suf])
        df[name] = make_metric(df, func, col=col, **kwargs)
        col_list.append(name)
    return col_list


def compute_runtime(row, start_idx=0, unit='sec', scale='lin'):
    if unit == 'sec':
        factor = 1
    elif unit == 'min':
        factor = 1 / 60
    elif unit == 'hour':
        factor = 1 / 3600
    else:
        raise ValueError('Unknown unit')
    times = row['time'][start_idx:]
    if len(times) > 0:
        last_iter = row['iter'][-1]
        val = np.mean(times) * last_iter * factor
        if scale == 'log':
            val = np.log10(val)
        return val
    else:
        return np.nan


def make_runtime_col(df, suf_col_dict, **kwargs):
    col_list = list()
    for suf, _ in suf_col_dict.items():
        history_col = f"{suf}_history"
        runtime_col = f"runtime_{kwargs.get('scale')}_{suf}"
        df[runtime_col] = df[history_col].apply(compute_runtime, **kwargs)
        col_list.append(runtime_col)
    return col_list


def compute_num_iter(row):
    return row['iter'][-1]


def make_num_iter(df, suf_col_dict):
    col_list = list()
    for suf, _ in suf_col_dict.items():
        history_col = f"{suf}_history"
        runtime_col = f"num_iter_{suf}"
        df[runtime_col] = df[history_col].apply(compute_num_iter)
        col_list.append(runtime_col)
    return col_list


def make_plot_df(df, suf_col_dict, agg_col, threshold=THRESHOLD):
    # Compute all desired performance metrices
    cols_acc = make_perf(df, metrics.accuracy, prefix='acc',
                         suf_col_dict=suf_col_dict, threshold=threshold)
    cols_f1score = make_perf(df, metrics.fscore, prefix='f1score',
                             suf_col_dict=suf_col_dict, threshold=threshold)
    cols_fp = make_perf(df, metrics.false_positive, prefix='fp',
                        suf_col_dict=suf_col_dict, threshold=threshold)
    cols_fn = make_perf(df, metrics.false_negative, prefix='fn',
                        suf_col_dict=suf_col_dict, threshold=threshold)
    cols_relerr = make_perf(df, metrics.relerr, prefix='relerr',
                            suf_col_dict=suf_col_dict)

    col_precAt_list = list()
    for n in [5, 10, 20, 50, 100, 200]:
        col_precAt_list += make_perf(df, metrics.precision_at_n,
                                     prefix=f'precAt{n}',
                                     suf_col_dict=suf_col_dict, n=n)

    col_runtime_lin = make_runtime_col(df, suf_col_dict, start_idx=0, unit='min',
                                       scale='lin')
    col_runtime_log = make_runtime_col(df, suf_col_dict, start_idx=0, unit='min',
                                       scale='log')

    col_num_iter = make_num_iter(df, suf_col_dict)

    if agg_col is None:
        return df
    else:
        # Make plotting df
        required_cols = (cols_acc + cols_relerr + cols_f1score
                         + col_precAt_list + col_num_iter
                         + col_runtime_lin + col_runtime_log
                         + cols_fp + cols_fn + [agg_col])
        agg_funcs = ['min', 'max', 'mean', 'std', 'count']
        df_plot = df[required_cols].groupby(agg_col).agg(agg_funcs)
        return df_plot


def plotmat_sidebyside(mats, labels=None, vmin=None, vmax=None, figsize=(5.5, 1.95), cmap_name="plasma", grid=None, ticks=None, ytitle=None, titlesize=None):
    if labels is None:
        assert isinstance(mats, dict)
        labels = list(mats.keys())
        mats = list(mats.values())

    if grid is not None:
        fig, axs = plt.subplots(*grid, figsize=figsize)
        axs = np.ravel(axs)
    elif len(mats) == 2:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    elif len(mats) == 3:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    elif len(mats) == 4:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = np.ravel(axs)
    elif len(mats) == 6:
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = np.ravel(axs)
    else:
        n = len(mats)
        num_cols = int(np.ceil(np.sqrt(n)))
        num_rows = int(np.floor(np.sqrt(n)))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        axs = np.ravel(axs)

        while len(mats) < len(axs):
            mats.append(None)
            labels.append(None)

    if (vmin is None) and (vmax is None):
        extend = 'neither'
    elif (vmin is not None) and (vmax is None):
        extend = 'min'
    elif (vmin is None) and (vmax is not None):
        extend = 'max'
    else:
        extend = 'both'

    vmin = min(map(lambda A: A.min(), mats)) if vmin is None else vmin
    vmax = max(map(lambda A: A.max(), mats)) if vmax is None else vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(mpl.cm.get_cmap(cmap_name))

    for ax, M, label in zip(axs, mats, labels):
        plt.sca(ax)
        if M is not None:
            ax.invert_yaxis()
            ax.set_aspect(1.0)
            dim = len(M)
            X = np.tile(np.arange(dim+1)+0.5, (dim+1,1))
            Y = X.T
            p = plt.pcolormesh(X, Y, M, norm=norm, cmap=cmap)
            if ticks:
                plt.xticks(ticks)
                plt.yticks(ticks)
            p.cmap.set_over('white')
            p.cmap.set_under('black')
            plt.title(label, pad=10, y=ytitle, )
        else:
            plt.axis('off')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(p, cax=cax, extend=extend)
    plt.show()
    return fig


def print_report(name, adj_hat, adj_true, thresh=0.05):
    adj_hat_flat = adj_hat.flatten()
    adj_true_flat = adj_true.flatten()

    # Relative error
    relerr = metrics.relerr(adj_hat_flat, adj_true_flat)

    # Accuracy
    acc = metrics.accuracy(adj_hat_flat, adj_true_flat, threshold=thresh)
    # Precimetrics
    prec = metrics.precision(adj_hat_flat, adj_true_flat, threshold=thresh)
    rec = metrics.recall(adj_hat_flat, adj_true_flat, threshold=thresh)
    fsc = metrics.fscore(adj_hat_flat, adj_true_flat, threshold=thresh)
    precat5 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=5)
    precat10 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=10)
    precat20 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=20)
    precat50 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=50)
    precat100 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=100)
    precat200 = metrics.precision_at_n(adj_hat_flat, adj_true_flat, n=200)
    # Error counts
    tp = metrics.tp(adj_hat_flat, adj_true_flat, threshold=thresh)
    fp = metrics.fp(adj_hat_flat, adj_true_flat, threshold=thresh)
    tn = metrics.tn(adj_hat_flat, adj_true_flat, threshold=thresh)
    fn = metrics.fn(adj_hat_flat, adj_true_flat, threshold=thresh)
    # Error rates
    tpr = metrics.tpr(adj_hat_flat, adj_true_flat, threshold=thresh)
    fpr = metrics.fpr(adj_hat_flat, adj_true_flat, threshold=thresh)
    tnr = metrics.tnr(adj_hat_flat, adj_true_flat, threshold=thresh)
    fnr = metrics.fnr(adj_hat_flat, adj_true_flat, threshold=thresh)

    import sklearn.metrics
    def roc_auc_score(adj_test, adj_true):
        return sklearn.metrics.roc_auc_score(np.ravel(adj_true) > 0, np.ravel(adj_test))
    def pr_auc_score(adj_test, adj_true):
        return sklearn.metrics.average_precision_score(np.ravel(adj_true) > 0, np.ravel(adj_test))
    pr_auc = pr_auc_score(adj_hat_flat, adj_true_flat)
    roc_auc = roc_auc_score(adj_hat_flat, adj_true_flat)

    print()
    print(f'========== Method: {name} ==========')
    print()
    print(f"Relative Error: {relerr:.2e} ({relerr:.2f})")
    print(f" PR-AUC: {pr_auc:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print()
    print(f"Accuracy: {acc:.2f}")
    print()
    print('Edge counts')
    print('------------')
    print(f"Pred: {np.sum(adj_hat_flat > thresh):.2f}")
    print(f"True: {np.sum(adj_true_flat > 0):.2f}")
    print()
    print('Error counts')
    print('------------')
    print(f" True Positive: {tp:.2f}")
    print(f"False Positive: {fp:.2f}")
    print(f" True Negative: {tn:.2f}")
    print(f"False Negative: {fn:.2f}")
    print()
    print('Error rates')
    print('-----------')
    print(f" True Positive Rate: {tpr:.2f}")
    print(f"False Positive Rate: {fpr:.2f}")
    print(f" True Negative Rate: {tnr:.2f}")
    print(f"False Negative Rate: {fnr:.2f}")
    print()
    print('F-Score')
    print('-------')
    print(f" F1-Score: {fsc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"   Recall: {rec:.2f}")
    print(f"   PR-AUC: {pr_auc:.2f}")
    print(f"  ROC-AUC: {roc_auc:.2f}")
    print()
    print('Precision@k')
    print('-----------')
    # print(f"  Prec@5: {precat5:.2f}")
    print(f" Prec@10: {precat10:.2f}")
    # print(f" Prec@20: {precat20:.2f}")
    print(f" Prec@50: {precat50:.2f}")
    # print(f"Prec@100: {precat100:.2f}")
    print(f"Prec@200: {precat200:.2f}")

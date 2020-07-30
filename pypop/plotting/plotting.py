import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from ..utils import metrics

THRESHOLD = 0.05  # Default threshold for small values


def set_notebook_config():
    SIGCONF_RCPARAMS = {
        "figure.autolayout": True,          # Makes sure nothing the feature is neat & tight.
        "figure.figsize": (5.5, 1.95),       # Column width: 3.333 in, space between cols: 0.333 in.
        "figure.dpi": 150,                  # Displays figures nicely in notebooks.
        "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
        "xtick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.5,
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",             # use serif rather than sans-serif
        "font.serif": "Linux Libertine O",  # use "Linux Libertine" as the standard font
        "font.size": 9,
        "axes.titlesize": 9,                # LaTeX default is 10pt font.
        "axes.labelsize": 7,                # LaTeX default is 10pt font.
        "legend.fontsize": 7,               # Make the legend/label fonts a little smaller
        "legend.frameon": False,            # Remove the black frame around the legend
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
        "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
        "pgf.preamble": [
            r'\usepackage{fontspec}',
            r'\usepackage{unicode-math}',
            r'\setmainfont{Linux Libertine O}',
            r'\setmathfont{Linux Libertine O}',
        ]
    }
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(SIGCONF_RCPARAMS)
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


def plotmat_sidebyside(A, B, C=None, label_A='', label_B='', label_C='', figsize=(5.5, 1.95)):
    if C is None:
        C = A
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 3, figsize=figsize)

    vmin = min(A.min(), B.min(), C.min())
    vmax = max(A.max(), B.max(), C.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'RdYlBu_r'

    plt.sca(axs[0])
    plt.imshow(A, norm=norm, cmap=cmap, interpolation='none')
    plt.title(label_A)
    plt.colorbar()

    plt.sca(axs[1])
    plt.imshow(B, norm=norm, cmap=cmap, interpolation='none')
    plt.title(label_B)
    plt.colorbar()

    if len(axs) == 3:
        plt.sca(axs[2])
        plt.imshow(C, norm=norm, cmap=cmap, interpolation='none')
        plt.title(label_C)
        plt.colorbar()

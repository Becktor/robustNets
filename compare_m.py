from utils import *
import seaborn as sns
import pandas as pd

sns.set()


def FAD():
    proportions = [0, 0.5, 0.75, 0.9]
    accuracy_log_v = {
        "No noise": [0.718],
        "25%": [.590],
        "50%": [.508]}

    df_v = pd.DataFrame(accuracy_log_v)
    df_v['Model'] = 'BL'
    accuracy_log_rw = {
        "No noise": [.699],
        "25%": [.658],
        "50%": [.593]}
    df_rw = pd.DataFrame(accuracy_log_rw)
    df_rw['Model'] = 'RW'

    accuracy_log_ra = {
        "No noise": [.692],
        "25%": [.671],
        "50%": [.603]}
    df_ra = pd.DataFrame(accuracy_log_ra)
    df_ra['Model'] = 'RA'
    return df_v, df_rw, df_ra, 'mAP', "FAD"


def SMD():
    proportions = [0, 0.5, 0.75, 0.9]
    accuracy_log_v = {
        "No noise": [.689],
        "25%": [.590],
        "50%": [.313]}

    df_v = pd.DataFrame(accuracy_log_v)
    df_v['Model'] = 'BL'
    accuracy_log_rw = {
        "No noise": [.717],
        "25%": [.623],
        "50%": [.592]}
    df_rw = pd.DataFrame(accuracy_log_rw)
    df_rw['Model'] = 'RW'

    accuracy_log_ra = {
        "No noise": [.710],
        "25%": [.637],
        "50%": [.608]}
    df_ra = pd.DataFrame(accuracy_log_ra)
    df_ra['Model'] = 'RA'
    return df_v, df_rw, df_ra, 'mAP', "SMD"


def compare_data_in_percent(df_ra, df_rw, df_v, metric, name):
    data = pd.concat([df_ra, df_rw, df_v]).set_index(['Model'])
    data_p = pd.DataFrame([1 - data["No noise"]['BL'] / data["No noise"], 1 - data["25%"]['BL'] / data["25%"],
                           1 - data["50%"]['BL'] / data["50%"]])
    data_p = data_p * 100
    data_p = data_p.drop('BL', axis=1)
    data_p = pd.melt(data_p.reset_index(), id_vars='index', value_name='accuracy')

    import matplotlib.ticker as mtick
    cpl = sns.color_palette(["#66c2a5", "#8da0cb"])
    sns.set_palette(cpl)
    with sns.axes_style(style='ticks'):
        g = sns.catplot(data=data_p, x='index', y='accuracy', hue='Model', hue_order=["RW", "RA"], kind="bar",
                        legend=False, height=5, aspect=6 / 5)
        g.set_axis_labels("Proportion", "Accuracy")
        plt.legend(loc="upper left")
        g.ax.set_title(f"{metric} comparison for {name} in %", size=16, y=1.02, x=0.45)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, left=0.12)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        ax = plt.gca()
        plt.plot([-.5, 0, 1, 2, 2.5], np.zeros(5), linewidth=1, color='black')
        for i, bar in enumerate(ax.patches):
            h = bar.get_height()
            font_size = 10
            if h < 0:
                ylim = ax.get_ylim()
                h -= (ylim[1] - ylim[0]) / (font_size * 2.5)
            c = 'black'
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    h,
                    f'{h:.2f}%',
                    fontsize=font_size,
                    color=c,
                    ha='center',
                    va='bottom')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.savefig(f'plots/{name}_{metric}_comp.png')
        plt.show()


def compare_best_in_percent(df_ra, df_rw, df_v, metric, name):
    data = pd.concat([df_ra, df_rw, df_v]).set_index(['Model'])
    data_p = pd.DataFrame([1 - data["No noise"]['BL'] / data["No noise"], 1 - data["No noise"]['BL'] / data["25%"],
                           1 - data["No noise"]['BL'] / data["50%"]])
    data_p = data_p * 100
    data_p = pd.melt(data_p.reset_index(), id_vars='index', value_name='accuracy')

    import matplotlib.ticker as mtick
    cpl = sns.color_palette(['#66c2a5', '#8da0cb','#fc8d62'])
    sns.set_palette(cpl)
    with sns.axes_style(style='ticks'):
        g = sns.catplot(data=data_p, x='index', y='accuracy', hue='Model', hue_order=["RW", "RA", "BL"], kind="bar",
                        legend=False, height=5, aspect=6 / 5)
        g.set_axis_labels("Proportion", "Accuracy")
        plt.legend(loc="lower left")
        g.ax.set_title(f"% Change from base-line model without noise", size=16, y=1.02, x=0.45)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, left=0.12)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        ax = plt.gca()
        plt.plot([-.5, 0, 1, 2, 2.5], np.zeros(5), linewidth=1, color='black')
        for i, bar in enumerate(ax.patches):
            h = bar.get_height()
            font_size = 10
            if h < 0:
                ylim = ax.get_ylim()
                h -= (ylim[1] - ylim[0]) / (font_size * 2.5)
            c = 'black'
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    h,
                    f'{h:.1f}%',
                    fontsize=font_size,
                    color=c,
                    ha='center',
                    va='bottom')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.savefig(f'plots/{name}_{metric}_comp_best.png')
        plt.show()





bl, rw, ra, name, metric = FAD()
compare_data_in_percent(bl, rw, ra, name, metric)
compare_best_in_percent(bl, rw, ra, name, metric)
bl, rw, ra, name, metric = SMD()
compare_data_in_percent(bl, rw, ra, name, metric)
compare_best_in_percent(bl, rw, ra, name, metric)
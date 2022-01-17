from utils import *
import seaborn as sns
import pandas as pd

sns.set()
import matplotlib.ticker as mtick

col = sns.color_palette(['#fc8d62', '#66c2a5', '#8da0cb'])
proportions = {0: "No noise",
               0.5: "50%",
               0.75: "75%",
               0.9: "90%"}
accuracy_log_v = {
    "No noise": [0.9105599641799926, 0.9188999772071839, 0.9010199785232544, 0.9209199786186218, 0.931499981880188],
    "50%": [0.8022199749946595, 0.8611999750137329, 0.7627199769020081, 0.8696399807929993, 0.8670599699020386],
    "75%": [0.2963199973106384, 0.5027799844741822, 0.5723599910736084, 0.6876999735832214, 0.6328999876976014],
    "90%": [0.3655999958515167, 0.40821999311447144, 0.4898399829864502, 0.47203998565673827, 0.396699982881546]}
df_v = pd.DataFrame(accuracy_log_v)
df_v['Model'] = 'BL'
# df_v['order'] = 0
accuracy_log_rw = {
    "No noise": [0.9539599776268005, 0.9392799735069275, 0.9552599787712097, 0.9476999759674072, 0.9525599718093872],
    "50%": [0.8764399886131287, 0.8892199754714966, 0.863919985294342, 0.8719399809837342, 0.8827399730682373],
    "75%": [0.8318599700927735, 0.8233999967575073, 0.8576599836349488, 0.8536799788475037, 0.8261799931526184],
    "90%": [0.7154599785804748, 0.704699981212616, 0.634879982471466, 0.7398799777030944, 0.7611199736595153]}
df_rw = pd.DataFrame(accuracy_log_rw)
df_rw['Model'] = 'RW'
# df_rw['order'] = 1
# accuracy_log_ra = {
# 0: [0.9411199808120727, 0.941379988193512, 0.9225399732589722, 0.938599967956543, 0.9380199790000916],
# 0.5: [0.8879399776458741, 0.8854399681091308, 0.8889999747276306, 0.8916399955749512, 0.8831399798393249],
# 0.75: [0.8216599702835083, 0.8562999725341797, 0.8628399729728699, 0.8364999651908874, 0.838379979133606],
#     0: [0.9423399686813354, 0.9352999806404114, 0.9420399785041809, 0.9430999755859375, 0.9517999768257142],
#     0.5: [0.8722799777984619, 0.8731599807739258, 0.8913199782371521, 0.8835799813270568, 0.8721399784088135],
#     0.75: [0.8534599900245666, 0.8495199680328369, 0.8671199917793274, 0.8578599691390991, 0.8402199864387512],
#     0.9: [0.7606799721717834, 0.7406199812889099, 0.7312999844551087, 0.7392799735069275, 0.7509799838066101]}

accuracy_log_ra = {
    "No noise": [0.9423399686813354, 0.9352999806404114, 0.9420399785041809, 0.9430999755859375, 0.9517999768257142],
    "50%": [0.8722799777984619, 0.8731599807739258, 0.8913199782371521, 0.8835799813270568, 0.8721399784088135],
    "75%": [0.8534599900245666, 0.8495199680328369, 0.8671199917793274, 0.8578599691390991, 0.8402199864387512],
    "90%": [0.7444799780845642, 0.783739972114563, 0.8018999934196472, 0.8026599884033203, 0.746239984035492]}
df_ra = pd.DataFrame(accuracy_log_ra)
df_ra['Model'] = 'RA'
# df_ra['order'] = 2
data = pd.concat([df_ra, df_rw, df_v], ignore_index=True)

plt.figure(figsize=(6, 5))
for i, prop in enumerate(proportions.keys()):
    acc = accuracy_log_v[proportions[prop]]
    plt.scatter([proportions[prop]] * len(accuracy_log_v[proportions[prop]]), acc, color=col[0], alpha=0.7)
    plt.scatter([proportions[prop]] * len(accuracy_log_rw[proportions[prop]]), accuracy_log_rw[proportions[prop]],
                color=col[1], alpha=0.7)
    plt.scatter([proportions[prop]] * len(accuracy_log_ra[proportions[prop]]), accuracy_log_ra[proportions[prop]],
                color=col[2], alpha=0.7)

# plot the trend line with error bars that correspond to standard deviation
order = ["No noise", "50%", "75%", "90%"]
accuracies_mean_v = np.array([np.mean(accuracy_log_v[k]) for k in order])
accuracies_std_v = np.array([np.std(accuracy_log_v[k]) for k in order])
plt.errorbar(proportions.values(), accuracies_mean_v, yerr=accuracies_std_v, label='BL', color=col[0],
             alpha=0.8, fmt='-o')

plt.fill_between(proportions.values(), accuracies_mean_v - accuracies_std_v,
                 accuracies_mean_v + accuracies_std_v,
                 interpolate=True, color=col[0], alpha=0.25)
print("v")
for x, y in zip(accuracies_mean_v, accuracies_std_v):
    print(f"{x:.3f}\pm{y:.3f}")



# RW
accuracies_mean_rw = np.array([np.mean(accuracy_log_rw[k]) for k in order])
accuracies_std_rw = np.array([np.std(accuracy_log_rw[k]) for k in order])
plt.errorbar(proportions.values(), accuracies_mean_rw, yerr=accuracies_std_rw, label='RW', color=col[1], alpha=0.8,
             fmt='-o')
plt.fill_between(proportions.values(), accuracies_mean_rw - accuracies_std_rw,
                 accuracies_mean_rw + accuracies_std_rw, interpolate=True, color=col[1], alpha=0.25)

print("rw")
for x, y in zip(accuracies_mean_rw, accuracies_std_rw):

    print(f"{x:.3f}\pm{y:.3f}")
##### RA
# plot the trend line with error bars that correspond to standard deviation
accuracies_mean_ra = np.array([np.mean(accuracy_log_ra[k]) for k in order])
accuracies_std_ra = np.array([np.std(accuracy_log_ra[k]) for k in order])

print("ra")
plt.errorbar(proportions.values(), accuracies_mean_ra, yerr=accuracies_std_ra, label='RA', color=col[2],
             alpha=0.8, fmt='-o')
for x, y in zip(accuracies_mean_ra, accuracies_std_ra):
    print(f"{x:.3f}\pm{y:.3f}")
plt.fill_between(proportions.values(), accuracies_mean_ra - accuracies_std_ra,
                 accuracies_mean_ra + accuracies_std_ra, interpolate=True, color=col[2], alpha=0.25)
plt.title('Accuracy of models on varying noise/data proportions', size=16, y=1.05)

plt.xlabel('Proportion')
plt.ylabel('Accuracy')
plt.legend(loc="upper right")
plt.ylim(top=0.96, bottom=0.35)
ax = plt.gca()
plt.savefig('mnist/ra_avg.png')
plt.show()

sns.set_palette(np.array(col)[1:])
data_m = data.groupby('Model').mean()
data_p = pd.DataFrame([1 - data_m["No noise"]['BL'] / data_m["No noise"], 1 - data_m["50%"]['BL'] / data_m["50%"],
                       1 - data_m["75%"]['BL'] / data_m["75%"], 1 - data_m["90%"]['BL'] / data_m["90%"]])
data_p = data_p * 100
data_p = data_p.drop('BL', axis=1)
data_p = pd.melt(data_p.reset_index(), id_vars='index', value_name='accuracy')

with sns.axes_style(style='ticks'):
    g = sns.catplot(data=data_p, x='index', y='accuracy', hue='Model', hue_order=["RW", "RA"], kind="bar",
                    legend=False, height=5, aspect=6 / 5)
    g.set_axis_labels("Proportion", "Accuracy in %")
    plt.legend(loc="upper left")
    ax = plt.gca()
    g.ax.set_title(f" % Increase compared to base-line model", size=16, y=1.02, x=0.45)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, left=0.12)
    for i, bar in enumerate(ax.patches):
        h = bar.get_height()
        c = 'black'  # col[i]
        ax.text(bar.get_x() + bar.get_width() / 1.9,
                bar.get_height(),
                f'{h:.1f}%',
                fontsize=10,
                color=c,
                ha='center',
                va='bottom')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.savefig('mnist/mnist_barplot_comparison.png')
    plt.show()

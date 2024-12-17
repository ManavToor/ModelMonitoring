# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script:

# dataset = pandas.DataFrame(0_x, 0_y, 1_x, 1_y, cur_ref, feature)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:

import matplotlib.pyplot as plt
import json

df = dataset

for _, row in df.iterrows():
    if row['cur_ref'] == 'ref':
        x0_ref = json.loads(row['x_0'])
        y0_ref = json.loads(row['y_0'])
        x1_ref = json.loads(row['x_1'])
        y1_ref = json.loads(row['y_1'])
    elif row['cur_ref'] == 'cur':
        x0_cur = json.loads(row['x_0'])
        y0_cur = json.loads(row['y_0'])
        x1_cur = json.loads(row['x_1'])
        y1_cur = json.loads(row['y_1'])

    feature = row['feature']

def plot_figure(x0, y0, x1, y1, ax, ref_cur, feat):
    categories = list(set(x0 + x1))
    category_to_index = {category: index for index, category in enumerate(categories)}
    val_0 = [category_to_index[category] for category in x0]
    val_1 = [category_to_index[category] for category in x1]

    bar_width = 1

    bars1 = ax.bar(x0, y0, color='#00A4AC', width=bar_width, label='0', alpha=0.7)
    bars2 = ax.bar(x1, y1, color='#1B3866', width=bar_width, label='1', alpha=0.7)

    # Highlight overlapping parts
    for bar1 in bars1:
        for bar2 in bars2:
            if bar1.get_x() == bar2.get_x():
                overlap_height = min(bar1.get_height(), bar2.get_height())
                ax.bar(bar1.get_x() + bar_width / 2, overlap_height, color='#0E6E89', width=bar_width)

    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.set_title(f'Target Occurance by {feat} ({ref_cur})')
    ax.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_figure(x0_ref, y0_ref, x1_ref, y1_ref, ax1, 'ref', feature)
plot_figure(x0_cur, y0_cur, x1_cur, y1_cur, ax2, 'cur', feature)

plt.tight_layout()
plt.show()
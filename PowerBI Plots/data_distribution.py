# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script:

# dataset = pandas.DataFrame(cur_x, ref_x, cur_y, ref_y)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:

import matplotlib.pyplot as plt
import json
import numpy as np

df = dataset

categories = json.loads(df.ref_x[0])
bar_width = 0.35
index = np.arange(len(json.loads(df.ref_x[0])))

fig, ax = plt.subplots()
bar2 = ax.bar(index - bar_width/2, json.loads(df.ref_y[0]), bar_width, label='cur', color='#00A4AC')
bar1 = ax.bar(index + bar_width/2, json.loads(df.cur_y[0]), bar_width, label='ref', color='#1B3866')

ax.set_title(f'Data Distribution of {df.column_name[0]}')
#ax.set_xticks(index)
ax.set_xticklabels(categories)
ax.set_ylabel('Count')
ax.set_xlabel('Values')
ax.legend()

plt.show()
# %%
# this file contains some code to get the ground truth for split tokens.

from datasets import load_dataset
import re
from tqdm import tqdm
from collections import Counter
import plotly.express as px
import pandas as pd

# %%
dataset = load_dataset("tdooms/TinyStories", split="train")

# %%
pattern = r'[^a-zA-Z\' ]'

counts = Counter(word for text in tqdm(dataset["text"]) for word in re.sub(pattern, " ", text.lower()).split(' '))

len(counts)
# %%

df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
df.reset_index(inplace=True)
df.columns = ['word', 'count']

df = df.sort_values(by='count', ascending=False)
df = df[df['count'] >= 50]
df = df.iloc[1:]
# %%

px.line(df, y='count', x ='word', title='Word Frequency')

# fig = px.bar(df, x='word', y='count', title='Word Frequency')
# fig.show()

# %%

df.to_csv("frequencies.csv", index=False)

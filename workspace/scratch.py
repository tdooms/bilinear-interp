# %%
import pandas as pd

df = pd.read_csv('old/datasets/classification.csv')
# %%
import plotly.express as px

px.histogram(df["kind"])

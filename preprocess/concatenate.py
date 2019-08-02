import pandas as pd
from functools import reduce
import os

path = "../test_data/E-GEOD-45642.processed.1"
buf = []
for file in os.listdir(path):
	buf.append(pd.read_csv(os.path.join(path, file), sep='\t'))


df = reduce(lambda df1, df2: pd.merge(df1, df2, on='ID_REF'), buf)
df.to_csv('../test_data/45642_conc.csv')
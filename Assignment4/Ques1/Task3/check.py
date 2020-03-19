import pandas as pd

doc = pd.read_pickle("collatz.pkl")
print (doc[12][1])

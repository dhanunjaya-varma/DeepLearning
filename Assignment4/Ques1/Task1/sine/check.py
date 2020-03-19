import pandas as pd

doc = pd.read_pickle("sin_tests.pkl")
print (doc[0][1])

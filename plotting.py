import pandas as pd

exp = 1554221367
path = 'Experiments/lambda_on_policy_control/{}/results.csv'.format(exp)
df = pd.read_csv(path)
df_sorted = df.sort_values('reward', ascending=False)

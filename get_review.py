import pandas as pd
df = pd.read_json('data/Cell_Phones_and_Accessories_5.json', lines=True)
row = df[['reviewText','overall']].sample(1, random_state=7).iloc[0]
print('Rating:', row['overall'])
print()
print(row['reviewText'])
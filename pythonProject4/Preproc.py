import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler


# read data from file
data = pd.read_csv('penguins.csv')

# get categorical data
categorical_data = data.select_dtypes(include=['object']).copy()

floating_data = data.select_dtypes(include=['float64']).copy()
integer_data = data.select_dtypes(include=['int64']).copy()

# get numerical data
numerical_data = pd.concat([floating_data , integer_data],axis=1)


# make normalization to the numerical data
scaler = MinMaxScaler()
normalized = scaler.fit_transform(numerical_data)
normalized_data = pd.DataFrame(normalized, columns=numerical_data.columns)


# Determine the species to three class Adelie to 1 Gentoo to 2 Chinstrap to 3
# and make the male 1 and female 2
replace_map = {'species': {'Adelie': 1, 'Gentoo': 0,'Chinstrap':-1 },
               'gender' : {'male':1, 'female':0}}


categorical_data_replace = categorical_data.copy()
categorical_data_replace.replace(replace_map, inplace=True)



new_data = pd.concat([categorical_data_replace , normalized_data] , axis=1)

# make all nan cell in gender column is male (1)
new_data['gender'] = new_data['gender'].fillna(1)

df=pd.DataFrame(new_data)
df.to_csv('PreprocessedData', index=False)






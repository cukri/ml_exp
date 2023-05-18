import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV

housing_path = r'F:\NAUKA\ML\ksiazka\project_chapter_2'
"""def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_path, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()"""


def load_housing_data():
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()


# print(housing)
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
# housing.hist(bins= 50, figsize=(20,15))
# plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()
housing_with_id["id"] = housing['longitude'] * 1000 + housing['latitude']
# train_set,test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])
housing['income_cat'].hist(bins=50, figsize=(20, 15))
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop(['income_cat'], axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population'] / 100, label='Populacja', figsize=(10, 7),
             c="median_house_value", colormap="jet", colorbar=True)

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=True))

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

housing['Pokoje_na_rodzine'] = housing['total_rooms']/housing['households']
housing['Sypialnie_na_pokoje'] = housing['total_bedrooms']/housing['total_rooms']
housing['Populacja_na_rodzine'] = housing['population']/housing['households']

housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

imputer = SimpleImputer(strategy='median')

housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)

housing_cat = housing[['ocean_proximity']]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAtrributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #żadnych zmiennych *args ani **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #Nie robi nic innego
    def transform(self, X):
        Pokoje_na_rodzine = X[:, rooms_ix] / X[:,households_ix]
        Populacja_na_rodzine = X[:,population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            Sypialnie_na_pokoje = X[:, bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,Pokoje_na_rodzine,Populacja_na_rodzine, Sypialnie_na_pokoje]
        else:
            return np.c_[X, Pokoje_na_rodzine, Populacja_na_rodzine]


attr_adder = CombinedAtrributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attr_adder', CombinedAtrributesAdder()),
    ('std_scaler', StandardScaler())
])

#housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])
print(housing.describe())

print(housing)
housing_prepared = full_pipeline.fit_transform(housing)

#wybor i uczenie modelu - regresja liniowa

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Progonzy:', lin_reg.predict(some_data_prepared))
print('etykiety:', list(some_labels))

#mierzenie błędu RMSE
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
"""
#uczenie modelu DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#sprawdzian krzyżowy - dzielenie zbioru uczacego na podzbiory trenujacy i walidacyjny. Uzywamy k-krotnego sprawdzianu
#krzyżowego(cross-validation)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
"""
def display_scores():
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("wyniki:", lin_rmse_scores)
    print("Średnia:", lin_rmse_scores.mean())
    print("Odchylenie_standardowe:", lin_rmse_scores.std())
    return float(lin_rmse_scores.mean())

#display_scores(tree_rmse_scores)
display_scores()

#uczenie modelu drzewa decyzyjnego
"""
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
print("ghiujk")
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)"""

#zapisywanie modelu w pliku
#joblib.dump(forest_reg, "forest_model.pkl")
#joblib.dump(lin_reg, "linear_model.pkl")
#joblib.dump(tree_reg, "tree_model.pkl")

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2,3,4]}
]
"""
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)"""
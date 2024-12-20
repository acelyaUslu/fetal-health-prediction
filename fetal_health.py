
#               FETAL HEALTH CLASSIFICATION
################################################################

#               PROBLEM
################################################################
# Cardiotocogram(CTG) muayenesinden elde edilen özellikler kullanılarak
# anne karnındaki bebeğin(fetal) sağlık durumunu sınıflandırmak
# için bir model geliştirilmek isteniyor.

#               DATASET
################################################################

# Bu veri seti, 2126 adet Cardiotocogram(CTG) muayenesinden elde edilen özelliklerin,
# anne karnındaki bebeğin (fetal) sağlık durumunu sınıflandırmak amacıyla
# üç uzman doğum uzmanı (obstetrist) tarafından
# Normal, Şüpheli ve Patolojik olmak üzere üç sınıfa ayrıldığı bir veri setidir.

#               FEATURES
################################################################

# baseline-value : Baseline Fetal Heart Rate
# accelerations : Number of accelerations per second
# fetal_movement : Number of fetal movements per second
# uterine_contractions : Number of uterine contractions per second
# light_decelerations : Number of LDs per second
# severe_decelerations : Number of SDs per second
# prolongued_decelerations : Number of PDs per second
# abnormal_short_term_variability : Percentage of time with abnormal short term variability
# mean_value_of_short_term_variability : Mean value of short term variability
# percentage_of_time_with_abnormal_long_term_variability : Percentage of time with abnormal long term variability
# mean_value_of_long_term_variability : Mean value of long term variability
# histogram_width : Width of the histogram made using all values from a record
# histogram_min : Histogram minimum value
# histogram_max : Histogram maximum value
# histogram_number_of_peaks : Number of peaks in the exam histogram
# histogram_number_of_zeroes : Number of zeroes in the exam histogram
# histogram_mode : Hist mode
# histogram_mean : Hist mean
# histogram_median : Hist Median
# histogram_variance : Hist variance
# histogram_tendency : Histogram trend
# fetal_health : Fetal health: 1 - Normal 2 - Suspect 3 - Pathological



#               IMPORTING LIBRARIES
################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import xgboost
# from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate, RandomizedSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# READING DATA SET
################################################################

df = pd.read_csv("datasets/fetal_health.csv")
df.head()


# EXPLORATORY DATA ANALYSIS(EDA)
################################################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)



### DEFINE CATEGORICAL and NUMERICAL FEATURES

def grab_col_names(dataframe, cat_th=10, car_th=25):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = de�i�ken say�s�.
    # num_but_cat cat_cols'un i�erisinde zaten.
    # dolay�s�yla t�m �u 3 liste ile t�m de�i�kenler se�ilmi� olacakt�r: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama i�in verilmi�tir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

## ANALYSIS of CATEGORICAL VARIABLES

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col,plot=True)

## ANALYSIS of NUMERICAL VARIABLES

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col,plot=True)


## ANALYSIS of TARGET VARIABLE

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"fetal_health",col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(numerical_col)[target].mean()}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "fetal_health", col)

## TARGET VARIABLE

df["fetal_health"].hist(bins=100)
plt.show(block=True)


## ANALYSIS OF CORELATION

corr = df[num_cols].corr()
corr

# SHOWING CORELATION

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)




# DATA PREPARATION
################################################################

# A-)Outlier Analysis
################################################################

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Outlier Control
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "fetal_health":
      print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "fetal_health":
        replace_with_thresholds(df,col)


# B-) Missing values
################################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


################################################################
# C-)Rare Analyser/Encoding
################################################################


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# rare class detection
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_analyser(df, "fetal_health", cat_cols)

rare_encoder(df,0.01)

################################################################
#                       CORRELATION
################################################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


high_correlated_cols(df, plot=True)

corr = df.corrwith(df['fetal_health']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,linewidths = 0.4,linecolor = 'black')
plt.title('Correlation w.r.t Fetal_health')


################################################################
#                           SCALING
################################################################

drop_list=["fetal_health"]
num_cols = [item for item in num_cols if item not in drop_list]


mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

df[num_cols] = ss.fit_transform(df[num_cols])

df[num_cols]
################################################################
#                   BASE MODELING
################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



y = df["fetal_health"]
X = df.drop("fetal_health", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_test,y_pred), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test, average='weighted'),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test, average='weighted'), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test, average='weighted'), 2)}")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)



from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=27)
smote_X_train, smote_y_train =  smote.fit_resample(X_train,y_train)

print("Before sampling class distribution:-",Counter(y_train))
print("After sampling class distribution:-",Counter(smote_y_train))

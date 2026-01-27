# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob, cmocean, matplotlib,rioxarray
import seaborn as sns
from scipy import stats
import geopandas as gpd
# all regression
from sklearn.linear_model import LinearRegression
# support vector regression
from sklearn.svm import SVR
# random forest regression
from sklearn.ensemble import RandomForestRegressor
# decision tree regression
from sklearn.tree import DecisionTreeRegressor
# KNN regression
from sklearn.neighbors import KNeighborsRegressor
# gradient boosting regression
from sklearn.ensemble import GradientBoostingRegressor
# AdaBoost regression
from sklearn.ensemble import AdaBoostRegressor
# XGBoost regression
from xgboost import XGBRegressor
# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
# 1d CNN
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.model_selection import train_test_split,cross_val_predict,KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# tensorflow model 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


DO= gpd.read_file(r'E:\my works\bloom\data\csv\DO.dbf')
#rename rename POINT_X and POINT_Y to lon and lat 
DO.rename(columns={'POINT_X':'lon','POINT_Y':'lat','RASTERVALU':'DO'},inplace=True)
# select lat lon and DO
DO=DO[['lon','lat','DO']]
DO

#
chla=pd.read_csv(r'E:\my works\bloom\data\csv\sampled_data_chla.csv')
# rename POINT_X and POINT_Y to lon and lat 
chla.rename(columns={'POINT_X':'lon','POINT_Y':'lat'},inplace=True)
# select lon,lat,chlor_a
chla=chla[['lon','lat','chlor_a']]
chla.head(1)

#sampled_data_poc.csv
poc=pd.read_csv(r'E:\my works\bloom\data\csv\sampled_data_poc.csv')
# rename POINT_X and POINT_Y to lon and lat
poc.rename(columns={'POINT_X':'lon','POINT_Y':'lat'},inplace=True)
# select lon,lat,poc
poc=poc[['lon','lat','poc']]
poc.head(1)

#
sst=pd.read_csv(r'E:\my works\bloom\data\csv\sampled_data_sst.csv')
# rename POINT_X and POINT_Y to lon and lat 
sst.rename(columns={'POINT_X':'lon','POINT_Y':'lat'},inplace=True)
# select lon and lat,sst
sst=sst[['lon','lat','sst']]
sst.head()

# merge sst,chla,poc
df=pd.merge(sst,chla,on=['lon','lat'])
df=pd.merge(df,poc,on=['lon','lat'])
# merge DO
df=pd.merge(df,DO,on=['lon','lat'])
df
#set working directory E:\my works\andaman\data\nc\copernicus
os.chdir(r'E:\my works\andaman\data\nc\copernicus')
# list all nc files
ncfiles=glob.glob('*.nc')
# split file name by _ and select 1st  
ncfiles=[i.split('_')[0] for i in ncfiles]
# unique file names
ncfiles=np.unique(ncfiles)
print(ncfiles)


# list all nc files
ncfiles=glob.glob('*.nc')
# select file if contains somxl030
#ncfiles=[i for i in ncfiles if 'somxl030' in i]#Mixed Layer Depth 0.03
# somxl010
#ncfiles=[i for i in ncfiles if 'somxl010' in i]#Mixed Layer Depth 0.01
#sosaline
ncfiles=[i for i in ncfiles if 'sosaline' in i]#Sea Surface Salinity
#sossheig
#ncfiles=[i for i in ncfiles if 'sossheig' in i]#Sea Surface Height
#sozotaux
#ncfiles=[i for i in ncfiles if 'sozotaux' in i]#Zonal Wind Stress
#sohtc300
#ncfiles=[i for i in ncfiles if 'sohtc300' in i]#Net Heat Flux at Sea Water Surface
# open first
ds=xr.open_dataset(ncfiles[0])
# Convert latitude and longitude to standard ranges
ds['x'] = (ds['x'] - 500) / 10
ds['y'] = ds['y']+180
# make a map sosaline
ds.sosaline.plot()
# dissolved inorganic carbon
ds=xr.open_dataset(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc')
# make above code as a function, input: nc file, variable name, depth, output: dataframe
def nc_to_df(ncfile, var):
    ds = xr.open_dataset(ncfile)
    ds = ds.sel(depth=4.940250e-01)
    df = pd.DataFrame()
    for i in range(len(sst)):
        ds1 = ds.sel(latitude=sst.iloc[i][1], longitude=sst.iloc[i][0], method='nearest')[var]
        dsvalues = ds1.to_dataframe()[var].values
        df = df.append(pd.DataFrame(dsvalues, columns=[var]))
    df.columns = [var]
    return df
#  apply to talk varible:total alkalinity
talk=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','talk')


# dissic: Dissolved inorganic carbon
dissic=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','dissic')
#no3: Nitrate
no3=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','no3')
#si: Silicate
si=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','si')
# nppv: Net Primary Production of Biomass by Phytoplankton
nppv=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','nppv')
# phyc: Phytoplankton Carbon
phyc=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','phyc')
# spco2: Surface Partial Pressure of CO2
spco2=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','spco2')
# o2: Dissolved Oxygen
o2=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','o2')
#po4: Phosphate
po4=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','po4')
#chl: Chlorophyll
chl=nc_to_df(r'E:\my works\bloom\data\NC\dissolved inorganic carbon\mercatorbiomer4v2r1_global_mean_20201201.nc','chl')

sst.reset_index(drop=True, inplace=True)
talk.reset_index(drop=True, inplace=True)
dissic.reset_index(drop=True, inplace=True)
no3.reset_index(drop=True, inplace=True)
si.reset_index(drop=True, inplace=True)
nppv.reset_index(drop=True, inplace=True)
phyc.reset_index(drop=True, inplace=True)
spco2.reset_index(drop=True, inplace=True)
o2.reset_index(drop=True, inplace=True)
po4.reset_index(drop=True, inplace=True)
chl.reset_index(drop=True, inplace=True)

df= pd.concat([sst.iloc[0:99026],talk.iloc[0:99026],dissic.iloc[0:99026], no3.iloc[0:99026],si.iloc[0:99026], nppv.iloc[0:99026],phyc.iloc[0:99026],spco2.iloc[0:99026],o2.iloc[0:99026],po4.iloc[0:99026],chl.iloc[0:99026]], axis=1)
df.dropna(inplace=True)

df = df.drop(['lat', 'lon'], axis=1)

# x features
x=df[['sst', 'talk', 'dissic', 'no3', 'si', 'nppv', 'phyc', 'spco2',
       'po4', 'chl']]
# y target
y=df['o2']
# split train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
# linear regression
lr=LinearRegression()
lr.fit(x_train,y_train)
#score
lr.score(x_test,y_test) #score means r2_score   
# define model
def model_score(model):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)
models=[LinearRegression(),SVR(),RandomForestRegressor(),DecisionTreeRegressor(),KNeighborsRegressor(),GradientBoostingRegressor(),AdaBoostRegressor(),XGBRegressor()]
model_names=['LinearRegression','SVR','RandomForestRegressor','DecisionTreeRegressor','KNeighborsRegressor','GradientBoostingRegressor','AdaBoostRegressor','XGBRegressor']

scores=[]
# Set a random seed for reproducibility
np.random.seed(42)
for model in range(len(models)):
    scores.append(model_score(models[model]))
# plot
plt.figure(figsize=(20,5))
plt.bar(model_names,scores)
# plot
plt.figure(figsize=(20,5))
plt.bar(model_names,scores)
# add percentage in bar
for i in range(len(scores)):
    plt.text(i,scores[i],str(round(scores[i]*100,2))+'%')
# split train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# Create dictionaries to store metrics
mse_scores = {}
rmse_scores = {}
r2_scores = {}

# Set a random seed for reproducibility
np.random.seed(42)
model_names=['LinearRegression','SVR','RandomForestRegressor','DecisionTreeRegressor','KNeighborsRegressor','GradientBoostingRegressor','AdaBoostRegressor','XGBRegressor']

# Iterate through models
for model_name, model in zip(model_names, models):
    # Fit the model
    model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate MSE, RMSE, and R²
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store scores in dictionaries
    mse_scores[model_name] = mse
    rmse_scores[model_name] = rmse
    r2_scores[model_name] = r2 

# Print the scores for each model
for model_name in model_names:
    print(f"{model_name}:")
    print(f"MSE: {mse_scores[model_name]}")
    print(f"RMSE: {rmse_scores[model_name]}")
    print(f"R-squared (R²): {r2_scores[model_name]}")
    print()
# Define the scores for each metric
mse_values = [mse_scores[model_name] for model_name in model_names]
rmse_values = [rmse_scores[model_name] for model_name in model_names]
r2_values = [r2_scores[model_name] for model_name in model_names]

# Create a bar plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set the bar width
bar_width = 0.25

# Define the x-axis positions for the bars
index = range(len(model_names))

# Create bars for MSE, RMSE, and R²
bar1 = ax.bar(index, mse_values, bar_width, label='MSE')
bar2 = ax.bar([i + bar_width for i in index], rmse_values, bar_width, label='RMSE')
bar3 = ax.bar([i + 2 * bar_width for i in index], r2_values, bar_width, label='R-squared')
# add text  mse_values, rmse_values, r2_values
for i in range(len(mse_values)):
    plt.text(i,mse_values[i],str(round(mse_values[i],2)))
    plt.text(i+bar_width,rmse_values[i],str(round(rmse_values[i],2)))
    plt.text(i+2*bar_width,r2_values[i],str(round(r2_values[i],2)))

# Set the x-axis labels
ax.set_xlabel('Regression Models')
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Set the y-axis label
ax.set_ylabel('Scores')

# Set the plot title
ax.set_title('Regression Model Performance Comparison')

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()   
# Define the number of folds for cross-validation
n_folds = 5  # You can adjust this based on your preference

# Create a KFold cross-validation object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

# Initialize lists to store metric scores for each fold
mse_scores = []
rmse_scores = []
r2_scores = []

# Perform cross-validation
for train_idx, val_idx in kf.split(x_train, y_train):
    x_train_fold, y_train_fold = x_train.iloc[train_idx], y_train.iloc[train_idx]
    x_val_fold, y_val_fold = x_train.iloc[val_idx], y_train.iloc[val_idx]

    # Fit the model on the training fold
    model.fit(x_train_fold, y_train_fold, epochs=100, verbose=0)

    # Predict on the validation fold
    y_pred_fold = model.predict(x_val_fold)

    # Calculate MSE, RMSE, and R² for the fold
    mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
    rmse_fold = np.sqrt(mse_fold)
    r2_fold = r2_score(y_val_fold, y_pred_fold)

    # Append fold scores to the lists
    mse_scores.append(mse_fold)
    rmse_scores.append(rmse_fold)
    r2_scores.append(r2_fold)

# Calculate the mean and standard deviation of scores across folds
mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)
mean_r2 = np.mean(r2_scores)

std_mse = np.std(mse_scores)
std_rmse = np.std(rmse_scores)
std_r2 = np.std(r2_scores)

# Print the average scores and their standard deviations
print(f"Mean MSE across {n_folds} folds: {mean_mse:.4f} (std: {std_mse:.4f})")
print(f"Mean RMSE across {n_folds} folds: {mean_rmse:.4f} (std: {std_rmse:.4f})")
print(f"Mean R-squared across {n_folds} folds: {mean_r2:.4f} (std: {std_r2:.4f})")


# define model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(x.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])
# compile model
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
# fit model
history = model.fit(x_train, y_train, epochs=100, validation_split = 0.2, verbose=0)
# plot loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
# print score
model.evaluate(x_test,y_test)
# Assuming you have already trained your model and have predictions
y_pred = model.predict(x_test)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Print the values
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")


# split train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# Standardize your feature data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape your input data
x_train = x_train[:, :, np.newaxis]  # Add a new axis to match the Conv1D input shape
x_test = x_test[:, :, np.newaxis]

# Build your 1D CNN model
model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Use mean squared error for regression

# Train the model
epochs = 50
batch_size = 32
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
# plot loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f"Mean Squared Error on Test Data: {loss}")
# Assuming you have already trained your model and have predictions
y_pred = model.predict(x_test)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Print the values
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")



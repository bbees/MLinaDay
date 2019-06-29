# Databricks notebook source
# MAGIC %md
# MAGIC # Book 1: Data Ingestion
# MAGIC This notebook executes the **first step of a data science process**, to download the raw data sets and store them as Spark Dataframes accessible to your Azure Databricks instance. 
# MAGIC 
# MAGIC To examine the SPARK data frames constructed, an optional 1a_raw_data_exploring notebook has been included in the repository and copied to your Azure Databricks Workspace. You must run this data ingestion notebook before running the exploration notebook cells. The exploration notebook details the simulated data sets we used for this predictive maintenance solution example.
# MAGIC 
# MAGIC ### Data sources
# MAGIC The common data elements for predictive maintenance problems can be summarized as follows:
# MAGIC 
# MAGIC **Machine features:** The features specific to each individual machine, e.g. engine size, make, model, location, installation date.
# MAGIC Telemetry data: The operating condition data collected from sensors, e.g. temperature, vibration, operating speeds, pressures.
# MAGIC Maintenance history: The repair history of a machine, e.g. maintenance activities or component replacements, this can also include error code or runtime message logs.
# MAGIC Failure history: The failure history of a machine or component of interest.
# MAGIC It is possible that failure history is contained within maintenance history, either as in the form of special error codes or order dates for spare parts. In those cases, failures can be extracted from the maintenance data. Additionally, different business domains may have a variety of other data sources that influence failure patterns which are not listed here exhaustively. These should be identified by consulting the domain experts when building predictive models.
# MAGIC 
# MAGIC Some examples of above data elements from use cases are:
# MAGIC 
# MAGIC * Machine conditions and usage: Flight routes and times, sensor data collected from aircraft engines, sensor readings from ATM transactions, train events data, sensor readings from wind turbines, elevators and connected cars.
# MAGIC 
# MAGIC * Machine features: Circuit breaker technical specifications such as voltage levels, geolocation or car features such as make, model, engine size, tire types, production facility etc.
# MAGIC 
# MAGIC * Failure history: fight delay dates, aircraft component failure dates and types, ATM cash withdrawal transaction failures, train/elevator door failures, brake disk replacement order dates, wind turbine failure dates and circuit breaker command failures.
# MAGIC 
# MAGIC * Maintenance history: Flight error logs, ATM transaction error logs, train maintenance records including maintenance type, short description etc. and circuit breaker maintenance records.
# MAGIC 
# MAGIC Given the above data sources, the two main data types we observe in predictive maintenance domain are temporal data and static data. Failure history, machine conditions, repair history, usage history are time series indicated by the timestamp of data collection. Machine and operator specific features, are more static, since they usually describe the technical specifications of machines or operator’s properties.
# MAGIC 
# MAGIC For this scenario, we use a relatively large-scale data to walk you through the main steps from data ingestion (this Jupyter notebook), feature engineering, model building, and model operationalization and deployment. The code for the entire process is written in PySpark and implemented using Jupyter notebooks within Azure Databricks. We use Azure Databricks scheduled notebooks to simulate creating failure predictions in batch scenarios.
# MAGIC 
# MAGIC ### Step 1: Data Ingestion
# MAGIC This data aquisiton notebook will download the simulated predictive maintenance data sets **from our GitHub data store.** We do some preliminary data cleaning and store the results as a Spark data frame on the Azure Cluster for use in the remaining notebook steps of this analysis.

# COMMAND ----------

## Setup our environment by importing required libraries

# Github has been having some timeout issues. This should fix the problem for this dataset.
import socket
socket.setdefaulttimeout(90)

import glob
import os
# Read csv file from URL directly
import pandas as pd

import urllib
from datetime import datetime
# Setup the pyspark environment
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download simulated data sets
# MAGIC We will be reusing the raw simulated data files from another tutorial. **The notebook automatically downloads these files stored at Microsoft/SQL-Server-R-Services-Samples GitHub site.**
# MAGIC 
# MAGIC The five data files are:
# MAGIC - machines.csv
# MAGIC - maint.csv
# MAGIC - errors.csv
# MAGIC - telemetry.csv
# MAGIC - failures.csv
# MAGIC 
# MAGIC There are 1000 machines of four different models. Each machine contains four components of interest, and four sensors measuring voltage, pressure, vibration and rotation. A controller monitors the system and raises alerts for five different error conditions. Maintenance logs indicate when something is done to the machine which does not include a component replacement. A failure is defined by the replacement of a component.
# MAGIC ![image](https://camo.githubusercontent.com/307d8bc01b57ea44dc7c366e3aa0167871aa74e7/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f417a7572652f4261746368537061726b53636f72696e67507265646963746976654d61696e74656e616e63652f6d61737465722f696d616765732f6d616368696e652e706e67)
# MAGIC 
# MAGIC This notebook does some preliminary data cleanup, creates summary graphics for each data set to verify the data downloaded correctly, and stores the resulting data sets in DBFS.

# COMMAND ----------

# The raw data is stored on GitHub here:
# https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/PredictiveMaintenanceModelingGuide/Data
# We access it through this URL:
basedataurl = "https://media.githubusercontent.com/media/Microsoft/SQL-Server-R-Services-Samples/master/PredictiveMaintenanceModelingGuide/Data/"

# We will store each of these data sets in DBFS.

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_data'
MAINT_DATA = 'maint_data'
ERROR_DATA = 'errors_data'
TELEMETRY_DATA = 'telemetry_data'
FAILURE_DATA = 'failure_data'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Machines data set
# MAGIC This simulation tracks a simulated set of 1000 machines over the course of a single year (2015).
# MAGIC 
# MAGIC This data set includes information about each machine: Machine ID, model type and age (years in service).

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "machines.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
machines = pd.read_csv(datafile)

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
mach_spark = spark.createDataFrame(machines, 
                                   verifySchema=False)

# Write the Machine data set to intermediate storage
mach_spark.write.mode('overwrite').saveAsTable(MACH_DATA)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Errors data set
# MAGIC The error log contains non-breaking errors recorded while the machine is still operational. These errors are not considered failures, though they may be predictive of a future failure event. The error datetime field is rounded to the closest hour since the telemetry data (loaded later) is collected on an hourly rate.

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "errors.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
errors = pd.read_csv(datafile, encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
error_spark = spark.createDataFrame(errors, 
                               verifySchema=False)

# Write the Errors data set to intermediate storage
error_spark.write.mode('overwrite').saveAsTable(ERROR_DATA)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Maintenance data set
# MAGIC The maintenance log contains both scheduled and unscheduled maintenance records. Scheduled maintenance corresponds with regular inspection of components, unscheduled maintenance may arise from mechanical failure or other performance degradations. A failure record is generated for component replacement in the case of either maintenance events. Because maintenance events can also be used to infer component life, the maintenance data has been collected over two years (2014, 2015) instead of only over the year of interest (2015).

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "maint.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
maint = pd.read_csv(datafile, encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
maint_spark = spark.createDataFrame(maint, 
                              verifySchema=False)

# Write the Maintenance data set to intermediate storage
maint_spark.write.mode('overwrite').saveAsTable(MAINT_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Telemetry data set
# MAGIC The telemetry time-series data consists of voltage, rotation, pressure, and vibration sensor measurements collected from each machines in real time. The data is averaged over an hour and stored in the telemetry logs.

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "telemetry.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
telemetry = pd.read_csv(datafile, encoding='utf-8')

# handle missing values
# define groups of features 
features_datetime = ['datetime']
features_categorical = ['machineID']
features_numeric = list(set(telemetry.columns) - set(features_datetime) - set(features_categorical))

# Replace numeric NA with 0
telemetry[features_numeric] = telemetry[features_numeric].fillna(0)

# Replace categorical NA with 'Unknown'
telemetry[features_categorical]  = telemetry[features_categorical].fillna("Unknown")

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
# This line takes about 9.5 minutes to run.
telemetry_spark = spark.createDataFrame(telemetry, verifySchema=False)

# Write the telemetry data set to intermediate storage
telemetry_spark.write.mode('overwrite').saveAsTable(TELEMETRY_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Failures data set
# MAGIC Failures correspond to component replacements within the maintenance log. Each record contains the Machine ID, component type, and replacement datetime. These records will be used to create the machine learning labels we will be trying to predict.

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "failures.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)

# Read into pandas
failures = pd.read_csv(datafile, encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
failures_spark = spark.createDataFrame(failures, 
                                       verifySchema=False)

# Write the failures data set to intermediate storage
failures_spark.write.mode('overwrite').saveAsTable(FAILURE_DATA)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC We have now stored the Raw data required for this Predictive Maintenance scenario as Spark data frames in the Azure Databricks instance. You can examine them in the Data panel accessible on the left. You should see the following five data sources:
# MAGIC 
# MAGIC 1. error_files
# MAGIC 2. machine_files
# MAGIC 3. maint_files
# MAGIC 4. telemetry_files
# MAGIC 5. failure_files
# MAGIC 
# MAGIC The .\notebooks\1a_raw data exploration notebooks does a preliminary data exploration on these data sets to help understand what we are working on. These data sets will be used in the next step .\notebooks\2a_feature_engineering notebook to generate the analysis data sets containing model features for our predictive maintenance machine learning model.

# COMMAND ----------

# MAGIC %md
# MAGIC # Book 1A: Raw data exploration
# MAGIC This notebook can be run after executing the 1_data_ingestion notebook. This notebook examines the SPARK data frames constructed in the previous notebook. Much of the text from the 1_data_ingestion notebook has been repeated here for convenience.
# MAGIC 
# MAGIC ### Data source
# MAGIC The common data elements for predictive maintenance problems can be summarized as follows:
# MAGIC 
# MAGIC * Machine features: The features specific to each individual machine, e.g. engine size, make, model, location, installation date.
# MAGIC * Telemetry data: The operating condition data collected from sensors, e.g. temperature, vibration, operating speeds, pressures.
# MAGIC * Maintenance history: The repair history of a machine, e.g. maintenance activities or component replacements, this can also include error code or runtime message logs.
# MAGIC * Failure history: The failure history of a machine or component of interest.
# MAGIC It is possible that failure history is contained within maintenance history, either as in the form of special error codes or order dates for spare parts. In those cases, failures can be extracted from the maintenance data. Additionally, different business domains may have a variety of other data sources that influence failure patterns which are not listed here exhaustively. These should be identified by consulting the domain experts when building predictive models.
# MAGIC 
# MAGIC Some examples of above data elements from use cases are:
# MAGIC 
# MAGIC **Machine conditions and usage:** Flight routes and times, sensor data collected from aircraft engines, sensor readings from ATM transactions, train events data, sensor readings from wind turbines, elevators and connected cars.
# MAGIC 
# MAGIC **Machine features:** Circuit breaker technical specifications such as voltage levels, geolocation or car features such as make, model, engine size, tire types, production facility etc.
# MAGIC 
# MAGIC **Failure history:** fight delay dates, aircraft component failure dates and types, ATM cash withdrawal transaction failures, train/elevator door failures, brake disk replacement order dates, wind turbine failure dates and circuit breaker command failures.
# MAGIC 
# MAGIC **Maintenance history:** Flight error logs, ATM transaction error logs, train maintenance records including maintenance type, short description etc. and circuit breaker maintenance records.
# MAGIC 
# MAGIC Given the above data sources, the two main data types we observe in predictive maintenance domain are temporal data and static data. Failure history, machine conditions, repair history, usage history are time series indicated by the timestamp of data collection. Machine and operator specific features, are more static, since they usually describe the technical specifications of machines or operator’s properties.
# MAGIC 
# MAGIC For this scenario, we use relatively large-scale data to walk the user through the main steps from data ingestion, feature engineering, model building, and model operationalization and deployment. The code for the entire process is written in PySpark and implemented using Jupyter notebooks.
# MAGIC 
# MAGIC ## Step 1A: Data exploration.
# MAGIC We do some preliminary data cleaning and verification, and generate some data exploration figures to help understand the data we will be working with in the remainder of this scenario.
# MAGIC 
# MAGIC **Note:** This notebook will take about 2-3 minutes to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

## Setup our environment by importing required libraries
# For creating some preliminary EDA plots.
# %matplotlib inline
import matplotlib.pyplot as plt

# Read csv file from URL directly
import pandas as pd
from ggplot import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_data'
MAINT_DATA = 'maint_data'
ERROR_DATA = 'errors_data'
TELEMETRY_DATA = 'telemetry_data'
FAILURE_DATA = 'failure_data'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load simulated data sets
# MAGIC We downloaded the simulated data files in the .\notebooks\1_data_ingestion notebook and stored the data as SPARK data frames in the five data sets:
# MAGIC * machines_files
# MAGIC * maint_files
# MAGIC * errors_files
# MAGIC * telemetry_files
# MAGIC * failures_files
# MAGIC 
# MAGIC There are 1000 machines of four different models. Each machine contains four components of interest, and four sensors measuring voltage, pressure, vibration and rotation. A controller monitors the system and raises alerts for five different error conditions. Maintenance logs indicate when something is done to the machine which does not include a component replacement. A failure is defined by the replacement of a component.
# MAGIC ![title](https://camo.githubusercontent.com/307d8bc01b57ea44dc7c366e3aa0167871aa74e7/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f417a7572652f4261746368537061726b53636f72696e67507265646963746976654d61696e74656e616e63652f6d61737465722f696d616765732f6d616368696e652e706e67)
# MAGIC 
# MAGIC This notebook does some preliminary data cleanup and creates summary graphics for each data set to verify the data downloaded correctly
# MAGIC ##### Machines data set
# MAGIC This simulation tracks a simulated set of 1000 machines over the course of a single year (2015).
# MAGIC 
# MAGIC This data set includes information about each machine: Machine ID, model type and age (years in service).
# MAGIC 
# MAGIC The following figure plots a histogram of the machines age colored by the specific model.

# COMMAND ----------

mach = spark.table(MACH_DATA) # spark.sql("SELECT * FROM " + MACH_DATA)

machines = mach.toPandas()
# one hot encoding of the variable model, basically creates a set of dummy boolean variablesplt.figure(figsize=(8, 6))

fig, ax = plt.subplots()

_, bins, _ = plt.hist([machines.loc[machines['model'] == 'model1', 'age'],
                       machines.loc[machines['model'] == 'model2', 'age'],
                       machines.loc[machines['model'] == 'model3', 'age'],
                       machines.loc[machines['model'] == 'model4', 'age']],
                       20, stacked=True, 
                      label=['model1', 'model2', 'model3', 'model4'])

plt.xlabel('Age (yrs)')
plt.ylabel('Count')
plt.legend()
display(fig)

# COMMAND ----------

display(mach)

# COMMAND ----------

# MAGIC %md
# MAGIC The figure shows how long the collection of machines have been in service. It indicates there are four model types, shown in different colors, and all four models have been in service over the entire 20 years of service. The machine age will be a feature in our analysis, since we expect older machines may have a different set of errors and failures then machines that have not been in service long.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Errors data set
# MAGIC The error log contains non-breaking errors recorded while the machine is still operational. These errors are not considered failures, though they may be predictive of a future failure event. The error datetime field is rounded to the closest hour since the telemetry data (loaded later) is collected on an hourly rate.
# MAGIC The following histogram details the distribution of the errors tracked in the log files.

# COMMAND ----------

errors = spark.table(ERROR_DATA)

# COMMAND ----------

#Quick plot to show structure.
display(errors)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Maintenance data set
# MAGIC The maintenance log contains both scheduled and unscheduled maintenance records. Scheduled maintenance corresponds with regular inspection of components, unscheduled maintenance may arise from mechanical failure or other performance degradations. A failure record is generated for component replacement in the case of either maintenance events. Because maintenance events can also be used to infer component life, the maintenance data has been collected over two years (2014, 2015) instead of only over the year of interest (2015).

# COMMAND ----------

maint = spark.table(MAINT_DATA)

# COMMAND ----------

# Quick plot to show structure
display(maint)

# COMMAND ----------

# MAGIC %md
# MAGIC The figure shows a histogram of component replacements divided into the four component types over the entire maintenance history. It looks like these four components are replaced at similar rates.
# MAGIC 
# MAGIC There are many ways we might want to look at this data including calculating how long each component type lasts, or the time history of component replacements within each machine. This will take some preprocess of the data, which we are delaying until we do the feature engineering steps in the next example notebook.
# MAGIC 
# MAGIC Next, we convert the errors data to a Spark dataframe, and verify the data types have converted correctly.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Telemetry data set
# MAGIC The telemetry time-series data consists of voltage, rotation, pressure, and vibration sensor measurements collected from each machines in real time. The data is averaged over an hour and stored in the telemetry logs.
# MAGIC 
# MAGIC Rather than plot 8.7 million data points, this figure plots a month of measurements for a single machine. This is representative of each feature repeated for every machine over the entire year of sensor data collection.

# COMMAND ----------

telemetry = spark.table(TELEMETRY_DATA).toPandas()
plt_data = telemetry.loc[telemetry['machineID'] == 1]
# format datetime field which comes in as string
plt_data['datetime'] = pd.to_datetime(plt_data['datetime'], format="%Y-%m-%d %H:%M:%S")

# Quick plot to show structure
plot_df = plt_data.loc[(plt_data['datetime'] >= pd.to_datetime('2015-02-01')) &
                       (plt_data['datetime'] <= pd.to_datetime('2015-03-01'))]

plt_data = pd.melt(plot_df, id_vars=['datetime', 'machineID'])

#pl = ggplot(aes(x="datetime", y="value", color = "variable", group="variable"), plt_data) +\
#    geom_line() +\
#    scale_x_date(labels=date_format('%m-%d')) +\
#    facet_grid('variable', scales='free_y')

#display(pl)

display(plt_data)

# COMMAND ----------

# MAGIC %md
# MAGIC The figure shows one month worth of telemetry sensor data for one machine. Each sensor is shown in it's own panel.
# MAGIC Next, we convert the errors data to a Spark dataframe, and verify the data types have converted correctly.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Failures data set
# MAGIC Failures correspond to component replacements within the maintenance log. Each record contains the Machine ID, component type, and replacement datetime. These records will be used to create the machine learning labels we will be trying to predict.
# MAGIC 
# MAGIC The following histogram details the distribution of the failure records obtained from failure log. This log was built originally from component replacements the maintenance log file.

# COMMAND ----------

failures = spark.table(FAILURE_DATA)

# COMMAND ----------

# Plot failures
#pl = ggplot(aes(x="failure"), failures.toPandas()) + geom_bar(fill="blue", color="black")
#display(pl)
display(failures)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC With these data sets, we next create the manipulate and transform this data into analysis data sets in the .\notebooks\2a_feature_engineering Jupyter notebook. This notebook does the feature engineering required for our predictive maintenance machine learning model.

# COMMAND ----------

# MAGIC %md
# MAGIC # Book 2A: Feature Engineering
# MAGIC This notebook executes the second step of a data science process, manipulate and transform the raw data sets into an analysis data set for constructing a machine learning model, or for the machine learning model to consume. You must run the 1_data_ingestion notebook before running this notebook.
# MAGIC 
# MAGIC The scenario is constructed as a pipeline flow, where each notebook is optimized to perform in a batch setting for each of the ingest, feature engineering, model building, model scoring operations. To accomplish this, this 2a_feature_engineering notebook is designed to be used to generate a general data set for any of the training, calibrate, test or scoring operations. In this scenario, we use a temporal split strategy for these operations, so the notebook parameters are used to set date range filtering. The notebook creates a labeled data set using the parameters start_date and to_date to select the time period for training. This data set is stored in the features_table specified. After this cell completes, you should see the dataset under the Databricks Data icon.
# MAGIC 
# MAGIC You can Run All cells, or use the Databricks CLI to create a Databricks Job to do the same process automatically.
# MAGIC 
# MAGIC To examine the SPARK analysis data sets constructed in this notebook, the optional 2a_feature_exploration notebook has been included in the repostiory and copied to your Azure Databricks Workspace. You must run this engineering notebook before running the exploration notebook, which details the feature dataset created here for use in this predictive maintenance solution example.
# MAGIC 
# MAGIC According to Wikipedia, Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive.
# MAGIC 
# MAGIC This Feature engineering notebook will load the data sets created in the Data Ingestion notebook (1_data_ingestion) from DBFS and combine them to create a single data set of features (variables) that can be used to infer a machines's health condition over time. The notebook steps through several feature engineering and labeling methods to create this data set for use in our predictive maintenance machine learning solution.
# MAGIC 
# MAGIC Note: This notebook will take less than a minute to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

## Setup our environment by importing required libraries
import pyspark.sql.functions as F
import pyspark

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, round
from pyspark.sql.functions import datediff
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_data'
MAINT_DATA = 'maint_data'
ERROR_DATA = 'errors_data'
TELEMETRY_DATA = 'telemetry_data'
FAILURE_DATA = 'failure_data'

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("features_table","training_data")

dbutils.widgets.text("start_date", '2000-01-01')

dbutils.widgets.text("to_date", '2015-10-30')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering
# MAGIC Our feature engineering will combine the different data sources together to create a single data set of features (variables) that can be used to infer a machines's health condition over time. The ultimate goal is to generate a single record for each time unit within each asset. The record combines features and labels to be fed into the machine learning algorithm.
# MAGIC 
# MAGIC Predictive maintenance take historical data, marked with a timestamp, to predict current health of a component and the probability of failure within some future window of time. These problems can be characterised as a classification method involving time series data. Time series, since we want to use historical observations to predict what will happen in the future. Classification, because we classify the future as having a probability of failure.
# MAGIC 
# MAGIC #### Lag features
# MAGIC There are many ways of creating features from the time series data. We start by dividing the duration of data collection into time units where each record belongs to a single point in time for each asset. The measurement unit for is in fact arbitrary. Time can be in seconds, minutes, hours, days, or months, or it can be measured in cycles, miles or transactions. The measurement choice is typically specific to the use case domain.
# MAGIC 
# MAGIC Additionally, the time unit does not have to be the same as the frequency of data collection. For example, if temperature values were being collected every 10 seconds, picking a time unit of 10 seconds for analysis may inflate the number of examples without providing any additional information if the temperature changes slowly. A better strategy may be to average the temperature over a longer time horizon which might better capture variations that contribute to the target outcome.
# MAGIC 
# MAGIC Once we set the frequency of observations, we want to look for trends within measures, over time, in order to predict performance degradation, which we would like to connect to how likely a component will fail. We create features for these trends within each record using time lags over previous observations to check for these performance changes. The lag window size $W$ is a hyper parameter that we can optimize. The following figures indicate a rolling aggregate window strategy for averaging a measure $t_i$ over a window $W = 3$ previous observations. We are not constrained to averages, we can roll aggregates over counts, average, the standard deviation, outliers based on standard deviations, CUSUM measures, minimum and maximum values for the window.
# MAGIC 
# MAGIC ![title](https://camo.githubusercontent.com/4fd8cb01dcf535e45779ec88838e76f9aa06e960/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f417a7572652f4261746368537061726b53636f72696e67507265646963746976654d61696e74656e616e63652f6d61737465722f696d616765732f726f6c6c696e672d6167677265676174652d66656174757265732e706e67)
# MAGIC 
# MAGIC We could also use a tumbling window approach, if we were interested in a different time window measure than the frequncy of the observations. For example, we might have obersvations evert 6 or 12 hours, but want to create features aligned on a day or week basis.
# MAGIC 
# MAGIC ![title](https://camo.githubusercontent.com/66498174edfdc6b854b76a8e5fa00a530eb0a826/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f417a7572652f4261746368537061726b53636f72696e67507265646963746976654d61696e74656e616e63652f6d61737465722f696d616765732f74756d626c696e672d6167677265676174652d66656174757265732e706e67)
# MAGIC 
# MAGIC In the following sections, we will build our features using only a rolling strategy to demonstrate the process. We align our data, and then build features along those normalized observations times. We start with the telemetry data.
# MAGIC 
# MAGIC #### Telemetry features
# MAGIC Because the telemetry data set is the largest time series data we have, we start feature engineering here. The telemetry data has 8761000 hourly observations for out 1000 machines. We can improve the model performance by aligning our data by aggregating average sensor measures on a tumbling 12 hour window. In this case we replace the raw data with the tumbling window data, reducing the sensor data to 731000 observations. This will directly reduce the computaton time required to do the feature engineering, labeling and modeling required for our solution.
# MAGIC 
# MAGIC Once we have the reduced data, we set up our lag features by compute rolling aggregate measures such as mean, standard deviation, minimum, maximum, etc. to represent the short term history of the telemetry over time.
# MAGIC 
# MAGIC The following code blocks alignes the data on 12 hour observations and calculates a rolling mean and standard deviation of the telemetry data over the last 12, 24 and 36 hour lags.

# COMMAND ----------

TO_DATE = dbutils.widgets.get("to_date")
START_DATE = dbutils.widgets.get("start_date")
telemetry = spark.table(TELEMETRY_DATA).where(F.col("datetime") <= TO_DATE).where(F.col("datetime") > START_DATE).cache()

# rolling mean and standard deviation
# Temporary storage for rolling means
tel_mean = telemetry

# Which features are we interested in telemetry data set
rolling_features = ['volt','rotate', 'pressure', 'vibration']
      
# n hours = n * 3600 seconds  
time_val = 12 * 3600

# Choose the time_val hour timestamps to align the data
# dt_truncated looks at the column named "datetime" in the current data set.
# remember that Spark is lazy... this doesn't execute until it is in a withColumn statement.
dt_truncated = ((round(unix_timestamp(col("datetime")) / time_val) * time_val).cast("timestamp"))

# COMMAND ----------

# We choose windows for our rolling windows 12hrs, 24 hrs and 36 hrs
lags = [12, 24, 36]

# align the data
for lag_n in lags:
    wSpec = Window.partitionBy('machineID').orderBy('datetime').rowsBetween(1-lag_n, 0)
    for col_name in rolling_features:
        tel_mean = tel_mean.withColumn(col_name+'_rollingmean_'+str(lag_n), 
                                       F.avg(col(col_name)).over(wSpec))
        tel_mean = tel_mean.withColumn(col_name+'_rollingstd_'+str(lag_n), 
                                       F.stddev(col(col_name)).over(wSpec))

# Calculate lag values...
telemetry_feat = (tel_mean.withColumn("dt_truncated", dt_truncated)
                  .drop('volt', 'rotate', 'pressure', 'vibration')
                  .fillna(0)
                  .groupBy("machineID","dt_truncated")
                  .agg(F.mean('volt_rollingmean_12').alias('volt_rollingmean_12'),
                       F.mean('rotate_rollingmean_12').alias('rotate_rollingmean_12'), 
                       F.mean('pressure_rollingmean_12').alias('pressure_rollingmean_12'), 
                       F.mean('vibration_rollingmean_12').alias('vibration_rollingmean_12'), 
                       F.mean('volt_rollingmean_24').alias('volt_rollingmean_24'),
                       F.mean('rotate_rollingmean_24').alias('rotate_rollingmean_24'), 
                       F.mean('pressure_rollingmean_24').alias('pressure_rollingmean_24'), 
                       F.mean('vibration_rollingmean_24').alias('vibration_rollingmean_24'),
                       F.mean('volt_rollingmean_36').alias('volt_rollingmean_36'),
                       F.mean('vibration_rollingmean_36').alias('vibration_rollingmean_36'),
                       F.mean('rotate_rollingmean_36').alias('rotate_rollingmean_36'), 
                       F.mean('pressure_rollingmean_36').alias('pressure_rollingmean_36'), 
                       F.stddev('volt_rollingstd_12').alias('volt_rollingstd_12'),
                       F.stddev('rotate_rollingstd_12').alias('rotate_rollingstd_12'), 
                       F.stddev('pressure_rollingstd_12').alias('pressure_rollingstd_12'), 
                       F.stddev('vibration_rollingstd_12').alias('vibration_rollingstd_12'), 
                       F.stddev('volt_rollingstd_24').alias('volt_rollingstd_24'),
                       F.stddev('rotate_rollingstd_24').alias('rotate_rollingstd_24'), 
                       F.stddev('pressure_rollingstd_24').alias('pressure_rollingstd_24'), 
                       F.stddev('vibration_rollingstd_24').alias('vibration_rollingstd_24'),
                       F.stddev('volt_rollingstd_36').alias('volt_rollingstd_36'),
                       F.stddev('rotate_rollingstd_36').alias('rotate_rollingstd_36'), 
                       F.stddev('pressure_rollingstd_36').alias('pressure_rollingstd_36'), 
                       F.stddev('vibration_rollingstd_36').alias('vibration_rollingstd_36'), )).cache()

display(telemetry_feat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Errors features
# MAGIC Like telemetry data, errors come with timestamps. An important difference is that the error IDs are categorical values and should not be averaged over time intervals like the telemetry measurements. Instead, we count the number of errors of each type within a lag window.
# MAGIC 
# MAGIC Again, we align the error counts data by tumbling over the 12 hour window using a join with telemetry data.

# COMMAND ----------

errors = spark.table(ERROR_DATA).where(F.col("datetime") <= TO_DATE).where(F.col("datetime") > START_DATE).cache()

# create a column for each errorID 
error_ind = (errors.groupBy("machineID","datetime","errorID").pivot('errorID')
             .agg(F.count('machineID').alias('dummy')).drop('errorID').fillna(0)
             .groupBy("machineID","datetime")
             .agg(F.sum('error1').alias('error1sum'), 
                  F.sum('error2').alias('error2sum'), 
                  F.sum('error3').alias('error3sum'), 
                  F.sum('error4').alias('error4sum'), 
                  F.sum('error5').alias('error5sum')))

# join the telemetry data with errors
error_count = (telemetry.join(error_ind, 
                              ((telemetry['machineID'] == error_ind['machineID']) 
                               & (telemetry['datetime'] == error_ind['datetime'])), "left")
               .drop('volt', 'rotate', 'pressure', 'vibration')
               .drop(error_ind.machineID).drop(error_ind.datetime)
               .fillna(0))

error_features = ['error1sum','error2sum', 'error3sum', 'error4sum', 'error5sum']

wSpec = Window.partitionBy('machineID').orderBy('datetime').rowsBetween(1-24, 0)
for col_name in error_features:
    # We're only interested in the erros in the previous 24 hours.
    error_count = error_count.withColumn(col_name+'_rollingmean_24', 
                                         F.avg(col(col_name)).over(wSpec))

error_feat = (error_count.withColumn("dt_truncated", dt_truncated)
              .drop('error1sum', 'error2sum', 'error3sum', 'error4sum', 'error5sum').fillna(0)
              .groupBy("machineID","dt_truncated")
              .agg(F.mean('error1sum_rollingmean_24').alias('error1sum_rollingmean_24'), 
                   F.mean('error2sum_rollingmean_24').alias('error2sum_rollingmean_24'), 
                   F.mean('error3sum_rollingmean_24').alias('error3sum_rollingmean_24'), 
                   F.mean('error4sum_rollingmean_24').alias('error4sum_rollingmean_24'), 
                   F.mean('error5sum_rollingmean_24').alias('error5sum_rollingmean_24')))

display(error_feat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Days since last replacement from maintenance
# MAGIC A crucial data set in this example is the use of maintenance records, which contain the information regarding component replacement. Possible features from this data set can be the number of replacements of each component over time or to calculate how long it has been since a component has been replaced. Replacement time is expected to correlate better with component failures since the longer a component is used, the more degradation would be expected.
# MAGIC 
# MAGIC As a side note, creating lagging features from maintenance data is not straight forward. This type of ad-hoc feature engineering is very common in predictive maintenance as domain knowledge plays a crucial role in understanding the predictors of a failure problem. In the following code blocks, the days since last component replacement are calculated for each component from the maintenance data. We start by counting the component replacements for the set of machines.

# COMMAND ----------

maint = spark.table(MAINT_DATA).where(F.col("datetime") <= TO_DATE).where(F.col("datetime") > START_DATE).cache()

# create a column for each component replacement
maint_replace = (maint.groupBy("machineID","datetime","comp").pivot('comp')
                 .agg(F.count('machineID').alias('dummy')).fillna(0)
                 .groupBy("machineID","datetime")
                 .agg(F.sum('comp1').alias('comp1sum'), 
                      F.sum('comp2').alias('comp2sum'), 
                      F.sum('comp3').alias('comp3sum'),
                      F.sum('comp4').alias('comp4sum'))
                 .withColumnRenamed('datetime','datetime_maint')).cache()

display(maint_replace)

# COMMAND ----------

# We want to align the component information on telemetry features timestamps.
telemetry_times = (telemetry_feat.select(telemetry_feat.machineID, telemetry_feat.dt_truncated)
                   .withColumnRenamed('dt_truncated','datetime_tel')).cache()

# COMMAND ----------

def grab_component_records(self, telemetry_times, comp_sum="comp1sum", sincelastcomp_="sincelastcomp1",
                          comp_a='comp2sum', comp_b='comp3sum', comp_c='comp4sum'):
    maint_comp = (self.where(col(comp_sum) == '1').withColumnRenamed('datetime', 'datetime_maint')
                  .drop(comp_a, comp_b, comp_c)).cache()
    # Within each machine, get the last replacement date for each timepoint
    maint_tel_comp = (telemetry_times.join(maint_comp,
                                           ((telemetry_times['machineID'] == maint_comp['machineID'])
                                            & (telemetry_times['datetime_tel'] > maint_comp['datetime_maint'])
                                            & (maint_comp[(comp_sum)] == '1')))
                      .drop(maint_comp.machineID)).cache()
    # Calculate the number of days between replacements
    return (maint_tel_comp.withColumn(sincelastcomp_,
                                      datediff(maint_tel_comp.datetime_tel, maint_tel_comp.datetime_maint))
            .drop(maint_tel_comp.datetime_maint).drop(maint_tel_comp[comp_sum])).cache()

pyspark.sql.dataframe.DataFrame.grab_component_records = grab_component_records

# Grab component 1 records
comp1 = maint_replace.grab_component_records(telemetry_times, comp_sum="comp1sum", sincelastcomp_="sincelastcomp1",
                                            comp_a='comp2sum', comp_b='comp3sum', comp_c='comp4sum').cache()
comp2 = maint_replace.grab_component_records(telemetry_times, comp_sum="comp2sum", sincelastcomp_="sincelastcomp2",
                                            comp_a='comp1sum', comp_b='comp3sum', comp_c='comp4sum').cache()
comp3 = maint_replace.grab_component_records(telemetry_times, comp_sum="comp3sum", sincelastcomp_="sincelastcomp3",
                                            comp_a='comp1sum', comp_b='comp2sum', comp_c='comp4sum').cache()
comp4 = maint_replace.grab_component_records(telemetry_times, comp_sum="comp4sum", sincelastcomp_="sincelastcomp4",
                                            comp_a='comp1sum', comp_b='comp2sum', comp_c='comp3sum').cache()

# COMMAND ----------

# Join component 3 and 4
comp3_4 = (comp3.join(comp4, ((comp3['machineID'] == comp4['machineID']) 
                              & (comp3['datetime_tel'] == comp4['datetime_tel'])), "left")
           .drop(comp4.machineID).drop(comp4.datetime_tel)).cache()

# Join component 2 to 3 and 4
comp2_3_4 = (comp2.join(comp3_4, ((comp2['machineID'] == comp3_4['machineID']) 
                                  & (comp2['datetime_tel'] == comp3_4['datetime_tel'])), "left")
             .drop(comp3_4.machineID).drop(comp3_4.datetime_tel)).cache()

# Join component 1 to 2, 3 and 4
comps_feat = (comp1.join(comp2_3_4, ((comp1['machineID'] == comp2_3_4['machineID']) 
                                      & (comp1['datetime_tel'] == comp2_3_4['datetime_tel'])), "left")
               .drop(comp2_3_4.machineID).drop(comp2_3_4.datetime_tel)
               .groupBy("machineID", "datetime_tel")
               .agg(F.max('sincelastcomp1').alias('sincelastcomp1'), 
                    F.max('sincelastcomp2').alias('sincelastcomp2'), 
                    F.max('sincelastcomp3').alias('sincelastcomp3'), 
                    F.max('sincelastcomp4').alias('sincelastcomp4'))
               .fillna(0)).cache()

# Choose the time_val hour timestamps to align the data
dt_truncated = ((round(unix_timestamp(col("datetime_tel")) / time_val) * time_val).cast("timestamp"))

# Collect data
maint_feat = (comps_feat.withColumn("dt_truncated", dt_truncated)
              .groupBy("machineID","dt_truncated")
              .agg(F.mean('sincelastcomp1').alias('comp1sum'), 
                   F.mean('sincelastcomp2').alias('comp2sum'), 
                   F.mean('sincelastcomp3').alias('comp3sum'), 
                   F.mean('sincelastcomp4').alias('comp4sum'))).cache()

display(maint_feat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Machine features
# MAGIC The machine features capture specifics of the individuals. These can be used without further modification since it include descriptive information about the type of each machine and its age (number of years in service). If the age information had been recorded as a "first use date" for each machine, a transformation would have been necessary to turn those into a numeric values indicating the years in service.
# MAGIC 
# MAGIC We do need to create a set of dummy features, a set of boolean variables, to indicate the model of the machine. This can either be done manually, or using a one-hot encoding step. We use the one-hot encoding for demonstration purposes.

# COMMAND ----------

machines = spark.table(MACH_DATA).cache()

# one hot encoding of the variable model, basically creates a set of dummy boolean variables
catVarNames = ['model']  
sIndexers = [StringIndexer(inputCol=x, outputCol=x + '_indexed') for x in catVarNames]
machines_cat = Pipeline(stages=sIndexers).fit(machines).transform(machines)

# one-hot encode
ohEncoders = [OneHotEncoder(inputCol=x + '_indexed', outputCol=x + '_encoded')
              for x in catVarNames]

ohPipelineModel = Pipeline(stages=ohEncoders).fit(machines_cat)
machines_cat = ohPipelineModel.transform(machines_cat)

drop_list = [col_n for col_n in machines_cat.columns if 'indexed' in col_n]

machines_feat = machines_cat.select([column for column in machines_cat.columns if column not in drop_list]).cache()

display(machines_feat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merging feature data
# MAGIC Next, we merge the telemetry, maintenance, machine and error feature data sets into a large feature data set. Since most of the data has already been aligned on the 12 hour observation period, we can merge with a simple join strategy.

# COMMAND ----------

# join error features with component maintenance features
error_maint = (error_feat.join(maint_feat, 
                               ((error_feat['machineID'] == maint_feat['machineID']) 
                                & (error_feat['dt_truncated'] == maint_feat['dt_truncated'])), "left")
               .drop(maint_feat.machineID).drop(maint_feat.dt_truncated))

# now join that with machines features
error_maint_feat = (error_maint.join(machines_feat, 
                                     ((error_maint['machineID'] == machines_feat['machineID'])), "left")
                    .drop(machines_feat.machineID))

# Clean up some unecessary columns
error_maint_feat = error_maint_feat.select([c for c in error_maint_feat.columns if c not in 
                                            {'error1sum', 'error2sum', 'error3sum', 'error4sum', 'error5sum'}])

# join telemetry with error/maint/machine features to create final feature matrix
final_feat = (telemetry_feat.join(error_maint_feat, 
                                  ((telemetry_feat['machineID'] == error_maint_feat['machineID']) 
                                   & (telemetry_feat['dt_truncated'] == error_maint_feat['dt_truncated'])), "left")
              .drop(error_maint_feat.machineID).drop(error_maint_feat.dt_truncated))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Label construction
# MAGIC Predictive maintenance is supervised learning. To train a model to predict failures requires examples of failures, and the time series of observations leading up to those failures. Additionally, the model needs examples of periods of healthy operation in order to discern the difference between the two states. The classification between these states is typically a boolean label (healthy vs failed).
# MAGIC 
# MAGIC Once we have the healthy vs. failure states, the predictive maintenance approach is only useful if the method will give some advanced warning of an impending failure. To accomplish this prior warning criteria, we slightly modify the label definition from a failure event which occurs at a specific moment in time, to a longer window of failure event occurs within this window. The window length is defined by the business criteria. Is knowing a failure will occur within 12 hours, enough time to prevent the failure from happening? Is 24 hours, or 2 weeks? The ability of the model to accurately predict an impending failure is dependent sizing this window. If the failure signal is short, longer windows will not help, and can actually degrade, the potential performance.
# MAGIC 
# MAGIC To acheive the redefinition of failure to about to fail, we over label failure events, labeling all observations within the failure warning window as failed. The prediction problem then becomes estimating the probability of failure within this window. For this example scenerio, we estimate the probability that a machine will fail in the near future due to a failure of a certain component. More specifically, the goal is to compute the probability that a machine will fail in the next 7 days due to a component failure (component 1, 2, 3, or 4).
# MAGIC 
# MAGIC ![title](https://camo.githubusercontent.com/84fa609fc0213d169c856b15e962960569b01275/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f417a7572652f4261746368537061726b53636f72696e67507265646963746976654d61696e74656e616e63652f6d61737465722f696d616765732f6c6162656c6c696e672d666f722d62696e6172792d636c617373696669636174696f6e2e706e67)
# MAGIC 
# MAGIC Below, a categorical failure feature is created to serve as the label. All records within a 24 hour window before a failure of component 1 have failure="comp1", and so on for components 2, 3, and 4; all records not within 7 days of a component failure have failure="none".
# MAGIC 
# MAGIC The first step is to align the failure data to the feature observation time points (every 12 hours).

# COMMAND ----------

failures = spark.table(FAILURE_DATA).where(F.col("datetime") <= TO_DATE).where(F.col("datetime") > START_DATE) .cache()

# We need to redefine dt_truncated to align with the failures table
dt_truncated = ((round(unix_timestamp(col("datetime")) / time_val) * time_val).cast("timestamp"))

fail_diff = (failures.withColumn("dt_truncated", dt_truncated)
             .drop(failures.datetime))

# COMMAND ----------

# Next, we convert the labels from text to numeric values. In the end, this will transform the problem from boolean of 'healthy'/'impending failure' to a multiclass 'healthy'/'component n impending failure'.
# map the failure data to final feature matrix
labeled_features = (final_feat.join(fail_diff, 
                                    ((final_feat['machineID'] == fail_diff['machineID']) 
                                     & (final_feat['dt_truncated'] == fail_diff['dt_truncated'])), "left")
                    .drop(fail_diff.machineID).drop(fail_diff.dt_truncated)
                    .withColumn('failure', F.when(col('failure') == "comp1", 1.0).otherwise(col('failure')))
                    .withColumn('failure', F.when(col('failure') == "comp2", 2.0).otherwise(col('failure')))
                    .withColumn('failure', F.when(col('failure') == "comp3", 3.0).otherwise(col('failure')))
                    .withColumn('failure', F.when(col('failure') == "comp4", 4.0).otherwise(col('failure'))))

labeled_features = (labeled_features.withColumn("failure", 
                                                labeled_features.failure.cast(DoubleType()))
                    .fillna(0))

# COMMAND ----------

# To now, we have labels as failure events. To convert to impending failure, we over label over the previous 7 days as failed.

# COMMAND ----------

# lag values to manually backfill label (bfill =7)
my_window = Window.partitionBy('machineID').orderBy(labeled_features.dt_truncated.desc())

# Create the previous 7 days 
labeled_features = (labeled_features.withColumn("prev_value1", 
                                                F.lag(labeled_features.failure).
                                                over(my_window)).fillna(0))
labeled_features = (labeled_features.withColumn("prev_value2", 
                                                F.lag(labeled_features.prev_value1).
                                                over(my_window)).fillna(0))
labeled_features = (labeled_features.withColumn("prev_value3", 
                                                F.lag(labeled_features.prev_value2).
                                                over(my_window)).fillna(0))
labeled_features = (labeled_features.withColumn("prev_value4", 
                                                F.lag(labeled_features.prev_value3).
                                                over(my_window)).fillna(0)) 
labeled_features = (labeled_features.withColumn("prev_value5", 
                                                F.lag(labeled_features.prev_value4).
                                                over(my_window)).fillna(0)) 
labeled_features = (labeled_features.withColumn("prev_value6", 
                                                F.lag(labeled_features.prev_value5).
                                                over(my_window)).fillna(0))
labeled_features = (labeled_features.withColumn("prev_value7", 
                                                F.lag(labeled_features.prev_value6).
                                                over(my_window)).fillna(0))

# Create a label features
labeled_features = (labeled_features.withColumn('label', labeled_features.failure + 
                                                labeled_features.prev_value1 +
                                                labeled_features.prev_value2 +
                                                labeled_features.prev_value3 +
                                                labeled_features.prev_value4 +
                                                labeled_features.prev_value5 + 
                                                labeled_features.prev_value6 + 
                                                labeled_features.prev_value7))

# Restrict the label to be on the range of 0:4, and remove extra columns
labeled_features = (labeled_features.withColumn('label_e', F.when(col('label') > 4, 4.0)
                                                .otherwise(col('label')))
                    .drop(labeled_features.prev_value1).drop(labeled_features.prev_value2)
                    .drop(labeled_features.prev_value3).drop(labeled_features.prev_value4)
                    .drop(labeled_features.prev_value5).drop(labeled_features.prev_value6)
                    .drop(labeled_features.prev_value7).drop(labeled_features.label))

#target_table_to_drop = "dbfs:/user/hive/warehouse/" + dbutils.widgets.get("features_table") +"/"
dbutils.fs.rm(dbutils.widgets.get("features_table"),True)
labeled_features.write.mode('overwrite').saveAsTable(dbutils.widgets.get("features_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusion
# MAGIC We have now stored the features in an analysis data set required for this Predictive Maintenance scenario as Spark data frames in the Azure Databricks instance. You can examine them in the Data panel accessible on the left. The ./notebooks/2a_feature_exploration notebook does a preliminary data exploration on this data set to help understand what we are working on.
# MAGIC 
# MAGIC The next step is to build and compare machine learning models using the feature data set we have just created. The ./notebooks/2b_model_building notebook works through building either a Decision Tree Classifier and a Random Forest Classifier using this data set.

# COMMAND ----------

# MAGIC %md
# MAGIC # Book 2B: Model Building
# MAGIC This notebook constructs a machine learning model designed to predict component failure in the machine. You must run the 2_feature_engineering notebook before running this notebook.
# MAGIC 
# MAGIC You can either Run All cells, or use the Databricks CLI to create a Databricks Job to do the same process automatically.
# MAGIC 
# MAGIC To test the model constructed in this notebook, the 2b_model_testing notebook has been included in the repostiory and copied to your Azure Databricks Workspace. You must run this notebook before running the model testing notebook, which calculates some model performance metrics for this predictive maintenance model.
# MAGIC 
# MAGIC Using the labeled feature data set constructed in the 2a_feature_engineering Jupyter notebook, this notebook loads the data from the training data set. Then builds a machine learning model (either a decision tree classifier or a random forest classifier) to predict when different components within our machine population will fail. We store the model for deployment on the Databricks DBFS file system (dbfs:/storage/models/ + model_type + .pqt) in parquet format for use in testing (2b_model_testing) and scoring (3b_model_scoring) operations.
# MAGIC 
# MAGIC Note: This notebook will take about 2-4 minutes to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

# import the libraries
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# This is the final feature data file.
training_table= 'training_data'
model_type = 'RandomForest' # Use 'DecisionTree' or 'GBTClassifier' or 'RandomForest'

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("training_table",training_table)
dbutils.widgets.text("model", model_type)

# COMMAND ----------

# Load the training data.
train_data = spark.table(dbutils.widgets.get("training_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare the Training data
# MAGIC A fundamental practice in machine learning is to calibrate and test your model parameters on data that has not been used to train the model. Evaluation of the model requires splitting the available data into a training portion, a calibration portion and an evaluation portion. Typically, 80% of data is used to train the model and 10% each to calibrate any parameter selection and evaluate your model.
# MAGIC 
# MAGIC In general random splitting can be used, but since time series data have an inherent correlation between observations. For predictive maintenance problems, a time-dependent spliting strategy is often a better approach to estimate performance. For a time-dependent split, a single point in time is chosen, the model is trained on examples up to that point in time, and validated on the examples after that point. This simulates training on current data and score data collected in the future data after the splitting point is not known. However, care must be taken on labels near the split point. In this case, feature records within 7 days of the split point can not be labeled as a failure, since that is unobserved data.

# COMMAND ----------

# define list of input columns for downstream modeling

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = train_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

# COMMAND ----------

# Spark models require a vectorized data frame. We transform the dataset here and then split the data into a training and test set. We use this split data to train the model on 9 months of data (training data), and evaluate on the remaining 3 months (test data) going forward.
# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')
train_data = va.transform(train_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(train_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(train_data)

training = train_data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification models
# MAGIC A particular problem in predictive maintenance is machine failures are usually rare occurrences compared to normal operation. This is fortunate for the business as maintenance and saftey issues are few, but causes an imbalance in the label distribution. This imbalance leads to poor performance as algorithms tend to classify majority class examples at the expense of minority class, since the total misclassification error is much improved when majority class is labeled correctly. This causes low recall or precision rates, although accuracy can be high. It becomes a larger problem when the cost of false alarms is very high. To help with this problem, sampling techniques such as oversampling of the minority examples can be used. These methods are not covered in this notebook. Because of this, it is also important to look at evaluation metrics other than accuracy alone.
# MAGIC 
# MAGIC We will build and compare two different classification model approaches:
# MAGIC 
# MAGIC * **Decision Tree Classifier:** Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
# MAGIC 
# MAGIC * **Random Forest Classifier:** A random forest is an ensemble of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.
# MAGIC 
# MAGIC The next code block creates the model. You can choose between a DecisionTree or RandomForest by setting the 'model_type' variable. We have also included a series of model hyperparameters to guide your exploration of the model space.

# COMMAND ----------

model_type = dbutils.widgets.get("model")

# train a model.
if model_type == 'DecisionTree':
  model = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                 # Maximum depth of the tree. (>= 0) 
                                 # E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'
                                 maxDepth=15,
                                 # Max number of bins for discretizing continuous features. 
                                 # Must be >=2 and >= number of categories for any categorical feature.
                                 maxBins=32, 
                                 # Minimum number of instances each child must have after split. 
                                 # If a split causes the left or right child to have fewer than 
                                 # minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.
                                 minInstancesPerNode=1, 
                                 # Minimum information gain for a split to be considered at a tree node.
                                 minInfoGain=0.0, 
                                 # Criterion used for information gain calculation (case-insensitive). 
                                 # Supported options: entropy, gini')
                                 impurity="gini")

  ##=======================================================================================================================
  ## GBTClassifer is only valid for Binary Classifiers, this is a multiclass (failures 1-4) so no GBTClassifier
#elif model_type == 'GBTClassifier':
#  model = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
#                        maxIter=200, stepSize=0.1,
#                        maxDepth=15,
#                        maxBins=32, 
#                        minInstancesPerNode=1, 
#                        minInfoGain=0.0)
  ##=======================================================================================================================
else:
  model = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", 
                                      # Passed to DecisionTreeClassifier
                                      maxDepth=15, 
                                      maxBins=32, 
                                      minInstancesPerNode=1, 
                                      minInfoGain=0.0,
                                      impurity="gini",
                                      # Number of trees to train (>= 1)
                                      numTrees=200, 
                                      # The number of features to consider for splits at each tree node. 
                                      # Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].
                                      featureSubsetStrategy="sqrt", 
                                      # Fraction of the training data used for learning each  
                                      # decision tree, in range (0, 1].' 
                                      subsamplingRate = 0.632)

# chain indexers and model in a Pipeline
pipeline_cls_mthd = Pipeline(stages=[labelIndexer, featureIndexer, model])

# train model.  This also runs the indexers.
model_pipeline = pipeline_cls_mthd.fit(training)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Persist the model
# MAGIC Here we save the model in a parquet file on DBFS.

# COMMAND ----------

# save model
model_pipeline.write().overwrite().save("dbfs:/storage/models/" + model_type + ".pqt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC We have now stored the model on the Azure Databricks files system. The 2b_model_testing notebook tests the model on new data and calculates a set of model evaluation metrics to help us know how well the model may performa in a production setting.
# MAGIC 
# MAGIC The next step is to build the batch scoreing operations. The 3_Scoring_Pipeline notebook takes parameters to define the data to be scored, and using the model created here, calulates the probability of component failure in the machine population specified.

# COMMAND ----------

# MAGIC %md
# MAGIC # Book 2B: Model Testing
# MAGIC This notebook examines the model created in the 2b_model_building notebook.
# MAGIC 
# MAGIC Using the 2a_feature_engineering Jupyter notebook, this notebook creates a new test data set and scores the observations using the machine learning model (a decision tree classifier or a random forest classifier) created in the 2b_model_building to predict when different components within the test machine population will fail. Then using the known labels from the existing data, we calculate a set of evaluation metrics to understand how the model may perform when used in production settings.
# MAGIC 
# MAGIC **Note:** This notebook will take about 2-4 minutes to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

# import the libraries
# For some data handling
import numpy as np
import pandas as pd

from collections import OrderedDict

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

import matplotlib.pyplot as plt

# This is the final feature data file.
testing_table = 'testing_data'
model_type = 'RandomForest' # Use 'DecisionTree' or 'GBTClassifier' or 'RandomForest'

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("Testing_table",testing_table)
dbutils.widgets.text("Model", model_type)

dbutils.widgets.text("start_date", '2015-11-30')

dbutils.widgets.text("to_date", '2016-02-01')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the Training/Testing data
# MAGIC A fundamental practice in machine learning is to calibrate and test your model parameters on data that has not been used to train the model. Evaluation of the model requires splitting the available data into a training portion, a calibration portion and an evaluation portion. Typically, 80% of data is used to train the model and 10% each to calibrate any parameter selection and evaluate your model.
# MAGIC 
# MAGIC In general random splitting can be used, but since time series data have an inherent correlation between observations. For predictive maintenance problems, a time-dependent spliting strategy is often a better approach to estimate performance. For a time-dependent split, a single point in time is chosen, the model is trained on examples up to that point in time, and validated on the examples after that point. This simulates training on current data and score data collected in the future data after the splitting point is not known. However, care must be taken on labels near the split point. In this case, feature records within 7 days of the split point can not be labeled as a failure, since that is unobserved data.
# MAGIC 
# MAGIC In the following code blocks, we create a data set to test the model.

# COMMAND ----------

#print(spark.catalog.listDatabases())
spark.catalog.setCurrentDatabase("default")
exists = False
for tbl in spark.catalog.listTables():
  if tbl.name == dbutils.widgets.get("Testing_table"):
    exists = True
    break

# COMMAND ----------

if not exists:
  dbutils.notebook.run("2a_feature_engineering", 600, {"features_table": dbutils.widgets.get("Testing_table"), 
                                                       "start_date": dbutils.widgets.get("start_date"), 
                                                       "to_date": dbutils.widgets.get("to_date")})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification models
# MAGIC A particular problem in predictive maintenance is machine failures are usually rare occurrences compared to normal operation. This is fortunate for the business as maintenance and saftey issues are few, but causes an imbalance in the label distribution. This imbalance leads to poor performance as algorithms tend to classify majority class examples at the expense of minority class, since the total misclassification error is much improved when majority class is labeled correctly. This causes low recall or precision rates, although accuracy can be high. It becomes a larger problem when the cost of false alarms is very high. To help with this problem, sampling techniques such as oversampling of the minority examples can be used. These methods are not covered in this notebook. Because of this, it is also important to look at evaluation metrics other than accuracy alone.
# MAGIC 
# MAGIC We will build and compare two different classification model approaches:
# MAGIC 
# MAGIC * **Decision Tree Classifier:** Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
# MAGIC 
# MAGIC * **Random Forest Classifier:** A random forest is an ensemble of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.
# MAGIC 
# MAGIC The next code block loads the model.

# COMMAND ----------

model_pipeline = PipelineModel.load("dbfs:/storage/models/" + dbutils.widgets.get("Model") + ".pqt")

print("Model loaded")
model_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC To evaluate this model, we predict the component failures over the test data set. Since the test set has been created from data the model has not been seen before, it simulates future data. The evaluation then can be generalize to how the model could perform when operationalized and used to score new data.

# COMMAND ----------

test_data = spark.table(dbutils.widgets.get("Testing_table"))

# define list of input columns for downstream modeling

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = test_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

#input_features
# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')

# assemble features
test_data = va.transform(test_data).select('machineID','dt_truncated','label_e','features').cache()

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(test_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(test_data)

testing = test_data

print(testing.count())

# make predictions. The Pipeline does all the same operations on the test data
predictions = model_pipeline.transform(testing)

# Create the confusion matrix for the multiclass prediction results
# This result assumes a decision boundary of p = 0.5
conf_table = predictions.stat.crosstab('indexedLabel', 'prediction')
confuse = conf_table.toPandas()
confuse.head()

# COMMAND ----------

# MAGIC %md
# MAGIC The confusion matrix lists each true component failure in rows and the predicted value in columns. Labels numbered 0.0 corresponds to no component failures. Labels numbered 1.0 through 4.0 correspond to failures in one of the four components in the machine. As an example, the third number in the top row indicates how many days we predicted component 2 would fail, when no components actually did fail. The second number in the second row, indicates how many days we correctly predicted a component 1 failure within the next 7 days.
# MAGIC 
# MAGIC We read the confusion matrix numbers along the diagonal as correctly classifying the component failures. Numbers above the diagonal indicate the model incorrectly predicting a failure when non occured, and those below indicate incorrectly predicting a non-failure for the row indicated component failure.
# MAGIC 
# MAGIC When evaluating classification models, it is convenient to reduce the results in the confusion matrix into a single performance statistic. However, depending on the problem space, it is impossible to always use the same statistic in this evaluation. Below, we calculate four such statistics.
# MAGIC 
# MAGIC * **Accuracy:** reports how often we correctly predicted the labeled data. Unfortunatly, when there is a class imbalance (a large number of one of the labels relative to others), this measure is biased towards the largest class. In this case non-failure days.
# MAGIC Because of the class imbalance inherint in predictive maintenance problems, it is better to look at the remaining statistics instead. Here positive predictions indicate a failure.
# MAGIC 
# MAGIC * **Precision:** Precision is a measure of how well the model classifies the truely positive samples. Precision depends on falsely classifying negative days as positive.
# MAGIC 
# MAGIC * **Recall:** Recall is a measure of how well the model can find the positive samples. Recall depends on falsely classifying positive days as negative.
# MAGIC 
# MAGIC * **F1:** F1 considers both the precision and the recall. F1 score is the harmonic average of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# MAGIC 
# MAGIC These metrics make the most sense for binary classifiers, though they are still useful for comparision in our multiclass setting. Below we calculate these evaluation statistics for the selected classifier.

# COMMAND ----------

# select (prediction, true label) and compute test error
# select (prediction, true label) and compute test error
# True positives - diagonal failure terms 
tp = confuse['1.0'][1]+confuse['2.0'][2]+confuse['3.0'][3]+confuse['4.0'][4]

# False positves - All failure terms - True positives
fp = np.sum(np.sum(confuse[['1.0', '2.0','3.0','4.0']])) - tp

# True negatives 
tn = confuse['0.0'][0]

# False negatives total of non-failure column - TN
fn = np.sum(np.sum(confuse[['0.0']])) - tn

# Accuracy is diagonal/total 
acc_n = tn + tp
acc_d = np.sum(np.sum(confuse[['0.0','1.0', '2.0','3.0','4.0']]))
acc = acc_n/acc_d

# Calculate precision and recall.
prec = tp/(tp+fp)
rec = tp/(tp+fn)

# Print the evaluation metrics to the notebook
print("Accuracy = %g" % acc)
print("Precision = %g" % prec)
print("Recall = %g" % rec )
print("F1 = %g" % (2.0 * prec * rec/(prec + rec)))
print("")

# COMMAND ----------

importances = model_pipeline.stages[2]
x = range(34)

plt.close('all')

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

plt.bar(x, list(importances.featureImportances.values))
plt.xticks(x)
plt.xlabel('')
ax.set_xticklabels(input_features, rotation = 90, ha="left")
#plt.gcf().subplots_adjust(bottom=0.50)
#plt.tight_layout()
display(fig)
# input_features

# COMMAND ----------

# MAGIC %md
# MAGIC Remember that this is a simulated data set. We would expect a model built on real world data to behave very differently. The accuracy may still be close to one, but the precision and recall numbers would be much lower.
# MAGIC 
# MAGIC ### Conclusion
# MAGIC The next step is to build the batch scoreing operations. The 3b_model_scoring notebook takes parameters to define the data to be scored, and using the model created here, calulates the probability of component failure in the machine population specified.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3B: Model Scoring
# MAGIC Using a scoring data set constructed in the 2a_feature_engineering notebook, and a model constructed in the 2b_model_building notebook (this is run through the 2_Training_Pipeline notebook, this notebook loads the data and predicts the probability of component failure with the provided model.
# MAGIC 
# MAGIC This notebook can be run though the 3_Scoring_Pipeline notebook, which creates a temporary scoring data set before scoring the data with this notebook.
# MAGIC 
# MAGIC We provide the 3b_model_scoring_evalation notebook to examine the output of the scoring process.
# MAGIC 
# MAGIC **IMPORTANT NOTE** This notebook depends on there being a scoring_data set in the Databricks Data store. You can score any dataset constructed with the 2a_feature_engineering notebook, but you must specify that data set in the scoring_data parameter above, or this notebook will fail.
# MAGIC 
# MAGIC Note: This notebook should take less than a minute to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

# import the libraries
from pyspark.ml import PipelineModel
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

# The scoring uses the same feature engineering script used to train the model
scoring_table = 'testing_data'
results_table = 'results_output'

model = 'RandomForest' # Use 'DecisionTree' or 'RandomForest'

# COMMAND ----------

# Databricks parameters to customize the runs.
dbutils.widgets.removeAll()
dbutils.widgets.text("scoring_data", scoring_table)
dbutils.widgets.text("results_data", results_table)

dbutils.widgets.text("model", model)

# COMMAND ----------

# MAGIC %md
# MAGIC We need to run the feature engineering on the data we're interested in scoring (2a_feature_engineering). Spark MLlib models require a vectorized data frame. We transform the dataset here for model consumption. In a general scoring operation, we do not know the labels so we only need to construct the features.

# COMMAND ----------

sqlContext.refreshTable(dbutils.widgets.get("scoring_data")) 
score_data = spark.table(dbutils.widgets.get("scoring_data"))

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = score_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

input_features
# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')

# assemble features
score_data = va.transform(score_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(score_data)

# COMMAND ----------

# MAGIC %md
# MAGIC To evaluate this model, we predict the component failures over the test data set. Since the test set has been created from data the model has not been seen before, it simulates future data. The evaluation then can be generalize to how the model could perform when operationalized and used to score the data in real time.

# COMMAND ----------

# Load the model from local storage
model_pipeline = PipelineModel.load("dbfs:/storage/models/" + dbutils.widgets.get("model") + ".pqt")

# score the data. The Pipeline does all the same operations on this dataset
predictions = model_pipeline.transform(score_data)

#write results to data store for persistance.
predictions.write.mode('overwrite').saveAsTable(dbutils.widgets.get("results_data"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC We have provided an additional notebook (3a_model_scoring_evaluation) to examine how the process works.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3B: Model Scoring evaluation
# MAGIC Using the results data set constructed in the 3b_model_scoring Jupyter notebook, this notebook loads the data scores the observations.
# MAGIC 
# MAGIC **Note:** This notebook will take about 1 minutes to execute all cells, depending on the compute configuration you have setup.

# COMMAND ----------

# import the libraries

# For some data handling
import numpy as np
from pyspark.ml import PipelineModel
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

# The scoring uses the same feature engineering script used to train the model
results_table = 'results_output'

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("results_data", results_table)

# COMMAND ----------

# make predictions. The Pipeline does all the same operations on the test data
sqlContext.refreshTable(dbutils.widgets.get("results_data")) 
predictions =  spark.table(dbutils.widgets.get("results_data"))

# Create the confusion matrix for the multiclass prediction results
# This result assumes a decision boundary of p = 0.5
conf_table = predictions.stat.crosstab('indexedLabel', 'prediction')
confuse = conf_table.toPandas()
confuse.head()

# COMMAND ----------

# MAGIC %md
# MAGIC The confusion matrix lists each true component failure in rows and the predicted value in columns. Labels numbered 0.0 corresponds to no component failures. Labels numbered 1.0 through 4.0 correspond to failures in one of the four components in the machine. As an example, the third number in the top row indicates how many days we predicted component 2 would fail, when no components actually did fail. The second number in the second row, indicates how many days we correctly predicted a component 1 failure within the next 7 days.
# MAGIC 
# MAGIC We read the confusion matrix numbers along the diagonal as correctly classifying the component failures. Numbers above the diagonal indicate the model incorrectly predicting a failure when non occured, and those below indicate incorrectly predicting a non-failure for the row indicated component failure.
# MAGIC 
# MAGIC When evaluating classification models, it is convenient to reduce the results in the confusion matrix into a single performance statistic. However, depending on the problem space, it is impossible to always use the same statistic in this evaluation. Below, we calculate four such statistics.
# MAGIC 
# MAGIC * **Accuracy:** reports how often we correctly predicted the labeled data. Unfortunatly, when there is a class imbalance (a large number of one of the labels relative to others), this measure is biased towards the largest class. In this case non-failure days.
# MAGIC Because of the class imbalance inherint in predictive maintenance problems, it is better to look at the remaining statistics instead. Here positive predictions indicate a failure.
# MAGIC 
# MAGIC * **Precision:** Precision is a measure of how well the model classifies the truely positive samples. Precision depends on falsely classifying negative days as positive.
# MAGIC 
# MAGIC * **Recall:** Recall is a measure of how well the model can find the positive samples. Recall depends on falsely classifying positive days as negative.
# MAGIC 
# MAGIC * **F1:** F1 considers both the precision and the recall. F1 score is the harmonic average of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# MAGIC 
# MAGIC These metrics make the most sense for binary classifiers, though they are still useful for comparision in our multiclass setting. Below we calculate these evaluation statistics for the selected classifier.

# COMMAND ----------

# select (prediction, true label) and compute test error
# select (prediction, true label) and compute test error
# True positives - diagonal failure terms 
tp = confuse['1.0'][1]+confuse['2.0'][2]+confuse['3.0'][3]+confuse['4.0'][4]

# False positves - All failure terms - True positives
fp = np.sum(np.sum(confuse[['1.0', '2.0','3.0','4.0']])) - tp

# True negatives 
tn = confuse['0.0'][0]

# False negatives total of non-failure column - TN
fn = np.sum(np.sum(confuse[['0.0']])) - tn

# Accuracy is diagonal/total 
acc_n = tn + tp
acc_d = np.sum(np.sum(confuse[['0.0','1.0', '2.0','3.0','4.0']]))
acc = acc_n/acc_d

# Calculate precision and recall.
prec = tp/(tp+fp)
rec = tp/(tp+fn)

# Print the evaluation metrics to the notebook
print("Accuracy = %g" % acc)
print("Precision = %g" % prec)
print("Recall = %g" % rec )
print("F1 = %g" % (2.0 * prec * rec/(prec + rec)))
print("")

# COMMAND ----------

# MAGIC %md
# MAGIC Remember that this is a simulated data set. We would expect a model built on real world data to behave very differently. The accuracy may still be close to one, but the precision and recall numbers would be much lower.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC This concludes this scenario. You can modify these notebooks to customize your own use case solution.
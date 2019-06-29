# Databricks notebook source
# MAGIC %md
# MAGIC ### Step 2A: Feature Engineering
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
# MAGIC 
# MAGIC #### Feature engineering
# MAGIC Our feature engineering will combine the different data sources together to create a single data set of features (variables) that can be used to infer a machines's health condition over time. The ultimate goal is to generate a single record for each time unit within each asset. The record combines features and labels to be fed into the machine learning algorithm.
# MAGIC 
# MAGIC Predictive maintenance take historical data, marked with a timestamp, to predict current health of a component and the probability of failure within some future window of time. These problems can be characterised as a classification method involving time series data. Time series, since we want to use historical observations to predict what will happen in the future. Classification, because we classify the future as having a probability of failure.
# MAGIC 
# MAGIC ##### Lag features
# MAGIC There are many ways of creating features from the time series data. We start by dividing the duration of data collection into time units where each record belongs to a single point in time for each asset. The measurement unit for is in fact arbitrary. Time can be in seconds, minutes, hours, days, or months, or it can be measured in cycles, miles or transactions. The measurement choice is typically specific to the use case domain.
# MAGIC 
# MAGIC Additionally, the time unit does not have to be the same as the frequency of data collection. For example, if temperature values were being collected every 10 seconds, picking a time unit of 10 seconds for analysis may inflate the number of examples without providing any additional information if the temperature changes slowly. A better strategy may be to average the temperature over a longer time horizon which might better capture variations that contribute to the target outcome.
# MAGIC 
# MAGIC Once we set the frequency of observations, we want to look for trends within measures, over time, in order to predict performance degradation, which we would like to connect to how likely a component will fail. We create features for these trends within each record using time lags over previous observations to check for these performance changes. The lag window size $W$ is a hyper parameter that we can optimize. The following figures indicate a rolling aggregate window strategy for averaging a measure $t_i$ over a window $W = 3$ previous observations. We are not constrained to averages, we can roll aggregates over counts, average, the standard deviation, outliers based on standard deviations, CUSUM measures, minimum and maximum values for the window.
# MAGIC 
# MAGIC #### Rolling Aggregates
# MAGIC 
# MAGIC We could also use a tumbling window approach, if we were interested in a different time window measure than the frequncy of the observations. For example, we might have obersvations evert 6 or 12 hours, but want to create features aligned on a day or week basis.
# MAGIC In the following sections, we will build our features using only a rolling strategy to demonstrate the process. We align our data, and then build features along those normalized observations times. We start with the telemetry data.

# COMMAND ----------

# MAGIC %md
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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Days since last replacement from maintenance
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

# COMMAND ----------

# MAGIC %md
# MAGIC Replacement features are then created by tracking the number of days between each component replacement. We'll repeat these calculations for each of the four components and join them together into a maintenance feature table.
# MAGIC 
# MAGIC First component number 1 (comp1):

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merging feature data
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
# MAGIC ### Label construction
# MAGIC Predictive maintenance is supervised learning. To train a model to predict failures requires examples of failures, and the time series of observations leading up to those failures. Additionally, the model needs examples of periods of healthy operation in order to discern the difference between the two states. The classification between these states is typically a boolean label (healthy vs failed).
# MAGIC 
# MAGIC Once we have the healthy vs. failure states, the predictive maintenance approach is only useful if the method will give some advanced warning of an impending failure. To accomplish this prior warning criteria, we slightly modify the label definition from a failure event which occurs at a specific moment in time, to a longer window of failure event occurs within this window. The window length is defined by the business criteria. Is knowing a failure will occur within 12 hours, enough time to prevent the failure from happening? Is 24 hours, or 2 weeks? The ability of the model to accurately predict an impending failure is dependent sizing this window. If the failure signal is short, longer windows will not help, and can actually degrade, the potential performance.
# MAGIC 
# MAGIC To acheive the redefinition of failure to about to fail, we over label failure events, labeling all observations within the failure warning window as failed. The prediction problem then becomes estimating the probability of failure within this window. For this example scenerio, we estimate the probability that a machine will fail in the near future due to a failure of a certain component. More specifically, the goal is to compute the probability that a machine will fail in the next 7 days due to a component failure (component 1, 2, 3, or 4).
# MAGIC 
# MAGIC Data Labeling
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

# MAGIC %md
# MAGIC Next, we convert the labels from text to numeric values. In the end, this will transform the problem from boolean of 'healthy'/'impending failure' to a multiclass 'healthy'/'component n impending failure'.

# COMMAND ----------

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

# MAGIC %md
# MAGIC To now, we have labels as failure events. To convert to impending failure, we over label over the previous 7 days as failed.

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

labeled_features.write.mode('overwrite').saveAsTable(dbutils.widgets.get("features_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Conclusion
# MAGIC We have now stored the features in an analysis data set required for this Predictive Maintenance scenario as Spark data frames in the Azure Databricks instance. You can examine them in the Data panel accessible on the left. The ./notebooks/2a_feature_exploration notebook does a preliminary data exploration on this data set to help understand what we are working on.
# MAGIC 
# MAGIC The next step is to build and compare machine learning models using the feature data set we have just created. The ./notebooks/2b_model_building notebook works through building either a Decision Tree Classifier and a Random Forest Classifier using this data set.
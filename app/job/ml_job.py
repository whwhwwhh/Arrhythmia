"""
ml_job.py
~~~~~~~~~~
This Python module contains a binary classification job. It can be
submitted to a Spark cluster (or locally) using the 'spark-submit'
command found in the '/bin' directory of all Spark distributions
(necessary for running any Spark job, locally or otherwise). For
example, this example script can be executed as follows,
    $SPARK_HOME/bin/spark-submit \
    --master spark://localhost:7077 \
    --py-files packages.zip \
    --files configs/ml_config.json \
    jobs/ml_job.py
where packages.zip contains Python modules required by this ML job (in
this example it contains a class to provide access to Spark's logger),
which need to be made available to each executor process on every node
in the cluster; ml_config.json is a text file sent to the cluster,
containing a JSON object with all of the configuration parameters
required by the ml job; and, ml_job.py contains the Spark application
to be executed by a driver process on the Spark master node.
"""

from app.dependencies.spark import start_spark
from app.dependencies.prepare_data import load_arraythmia_data
from app.dependencies.feature_selection import normalise_data, select_features
from app.dependencies.model import Model
from pyspark.ml.linalg import Vectors

        

def main():
    """Main ML job script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='my_ml_job',
        files=['configs/ml_config.json'])
    
    
    # log that main ml job is starting
    log.warn('ml_job is up-and-running')
    
    data_path = '../data/arrhythmia.data'
    
    
    np_array, labels = load_arraythmia_data(data_path)
    print(np_array)
    #optional 
    np_array = normalise_data(np_array)

    list_tuples = []
    for i in range(0, len(np_array)):
        list_tuples.append((labels[i],Vectors.dense(np_array[i])))
        
    data = spark.createDataFrame(list_tuples, ["label", "features"])
    
    sparkData = select_features(data)
    model = Model('GBTClassifier')
    
    #train, test = df_selected.randomSplit([0.9, 0.1], seed=12345)
    split_point = int(0.1 * (len(np_array)))
    test = spark.createDataFrame(sparkData.collect()[:split_point])
    train = spark.createDataFrame(sparkData.collect()[split_point:])
    
    model.tune(train, test, 'app/model/result')

    # log the success and terminate Spark application
    log.warn('ml_job is finished')
    spark.stop()

# entry point for PySpark ETL application
if __name__ == '__main__':
    main()
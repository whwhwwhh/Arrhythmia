# PySpark binary classification Project

This document is designed to be train a binary classifictaion model using pyspark.
The method is designed support train and tune for 4 different classification models (number of models can be expaned). 

## ETL Project Structure

The basic project structure is as follows:

```bash
root/
 |-- configs/
 |   |-- ml_config.json
 |-- dependencies/
 |   |-- logging.py
 |   |-- spark.py
 |   |-- model_config.py
 |   |-- prepare_data.py
 |   |-- model.py
 |   |-- feature_selection.py
 |-- jobs/
 |   |-- ml_job.py
 |-- tests/
 |   |-- I have not gottime to build this part
 |   build_dependencies.sh
 |   packages.zip
 |   Pipfile
 |   Pipfile.lock
```
## Running the ML job

Assuming that the `$SPARK_HOME` environment variable points to your local Spark installation folder, then the ETL job can be run from the project's root directory using the following command from the terminal,

```bash
$SPARK_HOME/bin/spark-submit \
--master local[*] \
--packages 'com.somesparkjar.dependency:1.0.0' \
--py-files packages.zip \
--files configs/ml_config.json \
jobs/ml_job.py
```

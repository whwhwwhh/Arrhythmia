from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplitModel

lr = LogisticRegression(maxIter=20, featuresCol = 'selectedFeatures', labelCol = 'label')

#dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
pipeline = Pipeline(stages=[lr])

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

#paramGrid = ParamGridBuilder() \
 #   .addGrid(dt.maxDepth, [2,3, 4, 5, 6]) \
  #  .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5) 
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)
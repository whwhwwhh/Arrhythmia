from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder
from dependencies.model_config import classifictaion_models
from pyspark.ml.param import Param
import numpy as np
import os
from pyspark.ml.tuning import CrossValidatorModel
"""
this script describes the binary classification structure.
I designed a dynamic model loading methods. so user can select one of the model
that is recorded in the model config file. 
"""
def load_model(model_folder):
        if os.path.exists(model_folder):
            try:
                model = PipelineModel.load(model_folder)
            except Exception as e:
                model = CrossValidatorModel.load(model_folder)
            return model
        else:
            print('model does not exist')
            return None

class Model():
    def __init__(self, model_name):
        if model_name in classifictaion_models:
            self.model_name = model_name
            self.params = classifictaion_models[model_name]
            self.tuning_params = classifictaion_models[model_name]['tuning_params']
            self.model = None
            #load a dynamic model
            classToImport = __import__("pyspark.ml.classification", fromlist=[self.model_name])
            dynamic_class = getattr(classToImport, self.model_name)
            args = {}
            for params_name, value in self.params.items():
                if isinstance(value, (str, int, float)):
                    args[params_name] = value
            self.dynamic_model = dynamic_class(**args)
        else:
            raise NameError('Please select the a valid classification model!')
    
    def save(self, folder):
        if self.model:
            self.model.write().save(os.path.join(folder, self.model_name))
        else:
            print('no model to save!')
    
    def predict(self, test):
        return self.model.transform(test)
        
    def train(self, train, test):
        pipeline = Pipeline(stages=[self.model])
        model = pipeline.fit(train)
        evaluator = BinaryClassificationEvaluator()
        prediction = model.transform(test)
        print(evaluator.evaluate(prediction))
        self.model = model
        return model
            
    def tune(self, train, test, model_save_to, useCrossValidation=True, numFolds=5):
        paramGrid = ParamGridBuilder()
        for name, value in self.tuning_params.items():
            param = Param(parent=self.dynamic_model, name=name, doc='')
            paramGrid = paramGrid.addGrid(param, value)
        paramGrid = paramGrid.build()
        
        pipeline = Pipeline(stages=[self.dynamic_model])
        
        crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=numFolds) 
        cvModel = crossval.fit(train)
        print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
        
        evaluator = BinaryClassificationEvaluator()
        prediction = cvModel.transform(test)
        print(evaluator.evaluate(prediction))
        
        cvModel.write().save(os.path.join(model_save_to, 'CrossValidationModel'))
        return cvModel

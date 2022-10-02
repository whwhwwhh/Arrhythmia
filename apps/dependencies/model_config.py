
#set up tunning parameters for selected models 
"""
this function includes configuration to selected binary classfication models
model slecting and tuning can be simply done by changing hyperparmers in this file.
the given examples illustrated how to do that. 
users can add and delete tuning_params as their needs.
"""
classifictaion_models = {
    'LogisticRegression':
        {'maxIter':30, 
         'featuresCol':'selectedFeatures', 
         'labelCol':'label', 
         'regParam':0.1, 
         'tuning_params':{
             'maxIter':[10, 20, 30],
             'regParam': [0.1, 0.01]
         }}, 
    'DecisionTreeClassifier':{
        "maxDepth":4, 
        'featuresCol':'features', 
        'labelCol':'label', 
        'tuning_params':{
             'maxIter':[10, 20, 30],
             'maxDepth': [4, 6, 8]
         }
         }, 
    'RandomForestClassifier':{
        "numTrees":10, 
        'featuresCol':'features', 
        'labelCol':'label', 
        'tuning_params':{
             'maxIter':[10, 20, 30],
             'numTrees': [6, 8 ,10, 12]
         }
        }, 
    'GBTClassifier':{
        "maxDepth":4, 
        'featuresCol':'features', 
        'labelCol':'label', 
        'tuning_params':{
             'maxIter':[10, 20, 30],
             'maxDepth': [4, 6, 8]
         }
        }
    }

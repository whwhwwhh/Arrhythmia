from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import UnivariateFeatureSelector

def normalise_data(dataFrame, normalizer=None):


    if normalizer:
        normal = Normalizer(inputCol="features", outputCol="normFeatures", p=normalizer)
        return normal.transform(dataFrame)
    else:
        return dataFrame
    
def select_features(data):
    selector = UnivariateFeatureSelector(outputCol="selectedFeatures")
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(100)
    model = selector.fit(data)
    model.getFeaturesCol()
    return model.transform(data)
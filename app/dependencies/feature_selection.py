
from sklearn.preprocessing import normalize
from pyspark.ml.feature import UnivariateFeatureSelector

def normalise_data(data, normalizer=None):
    if normalizer:
        return normalize(data, norm=normalizer)
    else:
        return data
    
def select_features(data):
    selector = UnivariateFeatureSelector(outputCol="selectedFeatures")
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(100)
    model = selector.fit(data)
    model.getFeaturesCol()
    return model.transform(data)
import numpy as np

from skimage.io import imread
from skimage.transform import resize
import pkg_resources
import joblib
from Phase2.transformers import RGB2GrayTransformer, HogTransformer

clf = joblib.load(pkg_resources.resource_filename('Phase2', 'image_classification_model.joblib'))
grayify = joblib.load(pkg_resources.resource_filename('Phase2', 'grayify.joblib'))
hogify = joblib.load(pkg_resources.resource_filename('Phase2', 'hogify.joblib'))
scalify = joblib.load(pkg_resources.resource_filename('Phase2', 'scalify.joblib'))

dog = imread('Images/test_skull.jpg')

output_shape = (102, 136, 3)

dog = resize(dog, output_shape=output_shape)

X_test = np.array([dog])

X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)
y_pred = clf.predict(X_test_prepared)

print(y_pred)

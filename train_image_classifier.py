import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib

from Phase2.transformers import RGB2GrayTransformer, HogTransformer

dog = imread('Images/dog1.jpg')
cat = imread('Images/cat1.jpg')

output_shape = (102, 136, 3)

dog = resize(dog, output_shape=output_shape)
cat = resize(cat, output_shape=output_shape)

train_inputs = [dog, cat]
train_labels = ['dog', 'cat']

for i in range(0, 9):
    index = str(i)
    path = 'Images/middlefinger' + index + '.jpg'
    img = imread(path)
    img = resize(img, output_shape=output_shape)
    train_inputs.append(img)
    train_labels.append('middlefinger')

for i in range(0, 9):
    index = str(i)
    path = 'Images/skull' + index + '.jpg'
    img = imread(path)
    img = resize(img, output_shape=output_shape)
    train_inputs.append(img)
    train_labels.append('skull')

# print(train_labels)

X_train = np.array(train_inputs)
y_train = np.array(train_labels)

grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14),
    cells_per_block=(3, 3),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

# print(X_train_prepared.shape)

# clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
clf = svm.SVC(kernel='linear')
clf.fit(X_train_prepared, y_train)

joblib.dump(clf, 'image_classification_model.joblib')
joblib.dump(grayify, 'grayify.joblib')
joblib.dump(hogify, 'hogify.joblib')
joblib.dump(scalify, 'scalify.joblib')

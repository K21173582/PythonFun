from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()

# Set X as the input features and y as the target variable
X = iris.data[:, :4]
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create the SVM model with a linear kernel
model = SVC(kernel='linear')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)

# Visualize the classification results
fig = plt.figure()
ax = Axes3D(fig)

# Plot the data points
for i, target_name in enumerate(iris.target_names):
    ax.scatter(X_test[y_pred==i, 0], X_test[y_pred==i, 1], X_test[y_pred==i, 2], label=target_name)

# Plot the hyperplane
coef = model.coef_[0]
intercept = model.intercept_[0]
xx = np.linspace(4, 8)
yy = np.linspace(1, 5)
zz = np.linspace(0, 3)
z = lambda x,y: (-intercept-coef[0]*x-coef[1]*y) / coef[2]
ax.plot(xx, yy, z(xx,yy), linestyle='--', color='k', label='Hyperplane')

# Set the labels and title
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')
ax.set_title('SVM classification')

# Set the legend
ax.legend()

# Show the plot
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# train an SVM classifier on the training data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# predict the labels of the test data
y_pred = clf.predict(X_test)

# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# define the class labels
class_names = iris.target_names

# plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn

# Load data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create vectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Vectorize data
data = vectorizer.fit_transform(newsgroups.data)

# Create LDA model
lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50., random_state=0)

# Fit LDA model
lda.fit(data)

# Visualize topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, data, vectorizer, mds='tsne')
pyLDAvis.display(vis)


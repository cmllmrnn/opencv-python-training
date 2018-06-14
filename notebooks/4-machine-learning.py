import cv2
import numpy as np

from sklearn.decomposition import PCA
pca = PCA(n_components=100)

# PCA: MNIST - kNN
# Load the MNIST image
mnist = cv2.imread('../datasets/digits.png', 0)

# Split the images by equal 50 x 100
images = [np.hsplit(row, 100) for row in np.vsplit(mnist, 50)]
images = np.array(images, dtype=np.float32)

# Split the images into 50/50 train/test set
train_features = images[:, :50].reshape(-1, (20 * 20))
test_features = images[:, 50:100].reshape(-1, (20 * 20))

# Apply PCA
pca.fit(train_features)
train_features = pca.transform(train_features)
test_features = pca.transform(test_features)

# Create labels
k = np.arange(10)
train_labels = np.repeat(k, 250).reshape(-1, 1)
test_labels = train_labels.copy()

# Create kNN model
knn = cv2.ml.KNearest_create()
knn.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)

# Classify test results, use k = 3
ret, result, neighbors, dist = knn.findNearest(test_features, 3)

# Measure model accuracy
matches = np.equal(result, test_labels)

# Convert boolean to int
matches = matches.astype(np.int)

# Count the correct predictions
correct = np.count_nonzero(matches)

# Calculate the accuracy
accuracy = (correct * 100.0) / result.size

# Print the accuracy
print("Acccuracy: {}".format(accuracy))

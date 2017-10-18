import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm

pd.options.mode.chained_assignment = None
all_images = pd.read_csv("../input/train.csv")

print("Train set loaded...")
images = all_images.iloc[:, 1:]
labels = all_images.iloc[:, :1]
images[images > 0] = 1


print("Images converted to binary")
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())


print("Fitting model succeeded...")
print(clf.score(test_images, test_labels))
test_images = pd.read_csv("../input/test.csv")
test_images[test_images > 0] = 1


print("Test set loaded...")
test_labels = clf.predict(test_images)
image_id = range(1, len(test_labels) + 1)
output = pd.DataFrame({'ImageId': image_id, 'Label': test_labels})


output.to_csv("output.csv", index=False)
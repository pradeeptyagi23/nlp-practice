#pip install skikit-learn
#sklearn bag of word model
from sklearn.feature_extraction.text import CountVectorizer

train_x = ["I love the book", "This is a great book","The fit is great","I love the shoes"]

#Utilize CountVectorizer to transform the statements into vector representation
vectorizer = CountVectorizer()

#If a word appears more than once , the CountVectorizer will increment the count as it is non-binary by default.
#Example : "I love the book book"
#If you want make it binary then pass binary = True
vectorizer = CountVectorizer(binary=True)
vectors  = vectorizer.fit_transform(train_x)

#Prints the unique names exracted from the sentences
#['book' 'fit' 'great' 'is' 'love' 'shoes' 'the' 'this']
# print(vectorizer.get_feature_names_out())

#Prints n dimensional array, showing occurences of each of the those unique names present or absent within those sentences
# [[1 0 0 0 1 0 1 0]
#  [1 0 1 1 0 0 0 1]
#  [0 1 1 1 0 0 1 0]
#  [0 0 0 0 1 1 1 0]]
# print(vectors.toarray())

#Build a model that classifies the sentences into book related and clothing related.
class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"

train_x_vectors = vectorizer.fit_transform(train_x)

#Feeding the bag of word models with training utterances
#The more training utterances you feed, the more powerful will be the model in predicting the classification
train_y = [Category.BOOKS, Category.BOOKS,Category.CLOTHING,Category.CLOTHING]

# This uses linear svm classificaton from skikit-learn
# A support vector machine (SVM) is a supervised machine learning algorithm that classifies data
from sklearn import svm
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors,train_y)
# print(clf_svm)

#predict the classification from the sentence based on the training classification
#This should classify as CLOTHING
test_x = vectorizer.transform(['shoes are alright'])

#This should classify as BOOKS
test_x = vectorizer.transform(['I love reading'])
print(clf_svm.predict(test_x))
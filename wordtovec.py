#pip install spacy
# python -m spacy download en_core_web_md (embeddings model)
import spacy
nlp = spacy.load('en_core_web_md')

train_x = ["I love the book", "This is a great book","The fit is great","I love the shoes","I like to eat burger"]


#Classification class to classify data in the train_x
class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
    FOOD = "FOOD"

#Feeding the bag of word models with training utterances
#The more training utterances you feed, the more powerful will be the model in predicting the classification
train_y = [Category.BOOKS, Category.BOOKS,Category.CLOTHING,Category.CLOTHING,Category.FOOD]


#first convert the phases into vector representation
docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]
#This will print the vector representation of each of the word in the first statement
#print(docs[0].vector)

#use classifier to train the data.
#First feed the vector data to the SVM . So that it can predict related data.
from sklearn import svm
clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors,train_y)

# test_x = ["I love to wear"]
test_x = ["I like to eat sandwich"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
print(clf_svm_wv.predict(test_x_word_vectors))
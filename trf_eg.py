#pip install spacy-transformers
#python -m spacy download en_core_web_sm
#python -m spacy download en_trf_bertbaseuncased_lg

import spacy

nlp = spacy.load("en_core_web_trf")
# doc = nlp("Here is some text to encode")

train_x = ["I love the book", "This is a great book","check out the book","need to make a deposit"]


#Classification class to classify data in the train_x
class Category:
    BOOKS = "BOOKS"
    BANK = "BANK"

train_y = [Category.BOOKS, Category.BOOKS,Category.BOOKS,Category.BANK]

#first convert the phases into vector representation
docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

from sklearn import svm
clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors,train_y)

# test_x = ["I love to wear"]
test_x = ["I need to write a check"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
print(clf_svm_wv.predict(test_x_word_vectors))
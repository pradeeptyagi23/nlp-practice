#pip install nltk
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
phrase = "I like reading stories"

#Tokenize the words and then stem each word to normalize it.
words = word_tokenize(phrase)

stemmed_words = []
for word in words:
    stemmed_words.append(stemmer.stem(word))

#Prints read the book
# print(" ".join(stemmed_words))
# stemmer.stem(phrase)


## Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

phrase = "I like reading stories"

#Tokenize the words and then stem each word to normalize it.
words = word_tokenize(phrase)


lemmatized_words = []
for word in words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

# print(" ".join(lemmatized_words))


## Stopword removal
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# print(stop_words)


phrase = "Here is an example sentence demonstrating the removal of stopwords"

#First tokenize the sentence
words = word_tokenize(phrase)

# Add stop words in the list by checkin the list of stop_words
stripped_phrase = []
for word in words:
    if word not in stop_words:
        stripped_phrase.append(word)

#stripped_phrase now consists everything except stopwords
print(" ".join(stripped_phrase))

## Various other techniques(spell correction,sentiment and part of speech tagging)

# pip install textblob

from textblob import TextBlob

phrase = "this is an examplee"
tb_phrase = TextBlob(phrase)

#spell correct
print(tb_phrase.correct())

#part of speech tagging. (classifying words with nouns,verbs,etc.)
# python -m textblob.download_corpora

print(tb_phrase.tags)

#sentiment classification. To check if there is negativity within statement or positivity.
#This is depicted by the polarity value in the output/


#This will have negative polarity
phrase = "this book is bad"


#This will have polarity 0.0 as it is neither positive or negative
phrase = "this is a book"

#This will have polarity >0.0 as it is a positive sentiment
phrase = "this is a good book"

tb_phrase = TextBlob(phrase)
print(tb_phrase.sentiment)
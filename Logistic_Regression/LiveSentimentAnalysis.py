from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "
english_stop_words = stopwords.words('english')
# New data to predict
pr = input("Type your review : ")
reviews = [pr]
def removeWhiteSpace(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

def get_lemmatized_text(corpus):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

preprocess1 = removeWhiteSpace(reviews)
print (preprocess1)
preprocess2 = remove_stop_words(preprocess1)
print (preprocess2)
preprocess3 = get_stemmed_text(preprocess2)
print (preprocess3)
preprocess4 = get_lemmatized_text(preprocess3)
print (preprocess4)

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(preprocess4)
X = ngram_vectorizer.transform(preprocess4)

print (X)

sc = StandardScaler(with_mean=False)
X_train_sc = sc.fit(X)
X_train = X_train_sc.transform(X)

print (X_train.shape[1])
# apply the whole pipeline to data
final_model = joblib.load('sentiment_model.pkl')
pred = final_model.predict(X_train)
print (pred)
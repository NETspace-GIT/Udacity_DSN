import sys
import nltk
import re
import numpy as np
import pandas as pd
import pickle
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier




def load_data(database):
    database_name = 'sqlite:///' + database
    database_engine = create_engine(database_name)

    df = pd.read_sql_table('Disasters', con=database_engine)
    print(df.head())

    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)

    for urls in detected_urls:
        text = text.replace(urls, "urlplaceholder")
    

    tokenizer = RegexpTokenizer(r'\w+')
    tokens_found = tokenizer.tokenize(text)
    
   
    lemmatizer = WordNetLemmatizer()

    tokens = []
    clean_tokens=[]
    for tok in tokens_found:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        tokens.append(clean_tok)
        clean_tokens=tokens
    return clean_tokens

def build_model():
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_predicted = model.predict(X_test)
    print(classification_report(Y_test, y_predicted, target_names=category_names))
    #results
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])



def save_model(model, model_filepath):
    # """model is saved as a pickel file here"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
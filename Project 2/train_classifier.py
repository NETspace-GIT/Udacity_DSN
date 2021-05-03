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
    """
    Function to load data
    
    Arguments:
        database_filepath : path to SQLite db
    Output:
        X : feature DataFrame
        Y : label DataFrame
        category_names : used for data visualization (app)
    """
    database_name = 'sqlite:///' + database
    database_engine = create_engine(database_name)

    df = pd.read_sql_table('Disasters', con=database_engine)
    print(df.head())

    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text : list of text messages (english)
    Output:
        clean_tokens : tokenized text, clean for ML modeling
    """
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
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline.
    
    """
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
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model : Scikit ML Pipeline
        X_test : test features
        Y_test : test labels
        category_names : label names (multi-output)
    """
    y_predicted = model.predict(X_test)
    print(classification_report(Y_test, y_predicted, target_names=category_names))
    #results
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])



def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
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
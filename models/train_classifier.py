# import libraries
import pickle
import re
import sys
import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
lemmatizer = WordNetLemmatizer()
nltk.download(['wordnet','punkt','stopwords'])
stop_words = nltk.corpus.stopwords.words("english")

def load_data(database_filepath):
    '''
    gets the file path to tehSQLite db
    param database_filepath:  path to db
    :return: X: features
    :return: y: target
    :return: catogory names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disastertb', con=engine)
    X=df['message']
    Y=df.iloc[:,4:-1]
    category_names=Y.columns
    return X,Y, category_names


def tokenize(text):
    '''this method does the following
    1. normalizing all the words to lower size
    2. removes punctuations
    3. splits the words
    4. removes the stopwords like am,is,have,you,...
    5. lammetizes the words for example running-->run
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    # normalize case and remove punctuation
    tokens = word_tokenize(text)    # tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    # lemmatize andremove stop words
    return tokens


def build_model():
    '''

    :return: returns a model which is ready to be trained
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multiclf',MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {'multiclf__estimator__n_estimators': [10,150],
              'multiclf__estimator__max_depth': [30,60]}

    model = GridSearchCV(pipeline, parameters, cv=5)

    return model



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(0,Y_test.shape[1]):
        print('------------------------')
        print(Y_test.columns[i])
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
        print('%25s accuracy : %.2f' %(Y_test.columns[i], accuracy_score(Y_test.iloc[:,i], y_pred[:,i])))
    for i in range(0,Y_test.shape[1]):
        print('%25s accuracy : %.2f' %(Y_test.columns[i], accuracy_score(Y_test.iloc[:,i], y_pred[:,i])))
    pass


def save_model(model, model_filepath):
    '''
    This model stores the model in the file path
    :param model: model to be stored
    :param model_filepath: fil path to store the model
    :return:
    '''
    with open(model_filepath+'trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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



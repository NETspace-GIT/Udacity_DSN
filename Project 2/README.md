# Disaster Response Pipeline Project

data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data

## Files  
- ETLPipelinePreparation.ipynb    

    
    - performs the tasks required before the data is fed onto the machinelearning pipelines 
   
- ML Pipeline Preparation.ipynb
  - builds a classifier to classify the messages
     
- app
  - temples
    - go.html
    - master.html
- data
  -disaster_categories (contains the different categories in which the data is classified)
  -disaster_messages   (contains the different messages recorded)
  -Diasterresponse     (the database file)
  -process.py          (loads ,cleans and saves the data)
  
- models
  -train_classifier.py (builds the model to classify the data and optimizes it)

- README.md  
  -file that summarizes this repository  


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

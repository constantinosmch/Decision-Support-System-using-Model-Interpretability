This source code is intended for the final year project assignment for Constantinos Michaelides (cm01007).

To run the programme, the system requires all the necessary libraries that have been used throughout the lifecycle of the application. 

Once all libraries have been installed, the GUI.py file can be executed. This programme uses all the pre-trained models and explanations mentioned in the report. To re-train the models, sections which have been labelled from the ANN.py, CNN.py and decisiontree_randomforest_train.py files should be uncommented and then executed. 

The ANN_Explainer, CNN_Explainer and DecisionTree_RandomForest_Diagrams files all contain samples of their respective explanations. The Confusion_Matrices contain the confusion matrices of the decision tree, random forest, ANN and CNN classifiers. Whereas, the data folder contains both datasets and the models folder contains all the pre-trained models.

The GUI can be easily navigated once properly set up, the main menu will have 4 buttons 3 of which take the user to the respective menu and the other option terminates the system. To add symptoms the user needs to simply select the symptoms from the dropdown menu and select the add button. To generate a prediction and explanation the generate explanation button can be selected whereas the go button only generates the prediction. To generate an MRI prediction the user needs to go to the MRI menu and select the upload button where a new window explorer screen will appear allowing for the user to select the desired image. To generate the prediction and explanation for the MRI classification the user needs to select the show explanation button whereas, to generate only the prediction the user needs to select the go button.

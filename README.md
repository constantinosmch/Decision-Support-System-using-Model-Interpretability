# Decision-Support-System-using-Model-Interpretability
Machine learning algorithms are known for their predictive capabilities and have been proven in previous literature that expert systems which utilise machine learning algorithms improve the detection and identification of abnormalities in the medical field.


In this project, various classifiers have been explored with the aim to reduce the misdiagnosis rate in medical examinations performed by doctors or other medical experts. In addition, this project focused on model interpretability and producing explanations from the classifiers as research has shown that providing comprehendible explanations allows for a greater level of trust between the user and the classifier. Indeed, this project developed a range of machine learning algorithms from decision trees and random forests to artificial neural networks and convolutional neural networks. The decision tree and random forest classifiers have been chosen due to the fact that these classifiers are often considered as ‘white boxes’ which allows the user to view the decision process of the classifier, thus making these models easily interpretable. Whereas, the neural networks are often referred to as ‘black box’ systems due to their complex technology which in turn makes the comprehension of the decision process hard and counterintuitive. This project focuses on developing model interpretability for both white box and black box systems.


The finalised models are then integrated into a python-based system which are then tested using real-world scenarios in order to provide decision support to the medical experts. This is achieved by generating various diagnoses and producing the respective explanation. The python-based system allows users easily interact with the different algorithms in an expected manner.

## Running the Code
To run the programme, the system requires all the necessary libraries that have been used throughout the lifecycle of the application. 

Once all libraries have been installed, the GUI.py file can be executed. This programme uses all the pre-trained models and explanations mentioned in the report. To re-train the models, sections which have been labelled from the ANN.py, CNN.py and decisiontree_randomforest_train.py files should be uncommented and then executed. 

The ANN_Explainer, CNN_Explainer and DecisionTree_RandomForest_Diagrams files all contain samples of their respective explanations. The Confusion_Matrices contain the confusion matrices of the decision tree, random forest, ANN and CNN classifiers. Whereas, the data folder contains both datasets and the models folder contains all the pre-trained models.

The GUI can be easily navigated once properly set up, the main menu will have 4 buttons 3 of which take the user to the respective menu and the other option terminates the system. To add symptoms the user needs to simply select the symptoms from the dropdown menu and select the add button. To generate a prediction and explanation the generate explanation button can be selected whereas the go button only generates the prediction. To generate an MRI prediction the user needs to go to the MRI menu and select the upload button where a new window explorer screen will appear allowing for the user to select the desired image. To generate the prediction and explanation for the MRI classification the user needs to select the show explanation button whereas, to generate only the prediction the user needs to select the go button.
[]()
## Example Output
Original Image with Mask          |  Displays Only Important Information Based on Classification | Displays Reasons for (Green) and Reasons Against (Red) the Classification
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/constantinosmch/Decision-Support-System-using-Model-Interpretability/blob/main/CNN_Explainer/Figure_4.png)  |  ![](https://github.com/constantinosmch/Decision-Support-System-using-Model-Interpretability/blob/main/CNN_Explainer/Figure_5.png) |  ![](https://github.com/constantinosmch/Decision-Support-System-using-Model-Interpretability/blob/main/CNN_Explainer/Figure_6.png)
<!-- <span class="img_container center" style="display: block;"> -->
<!--     <img alt="test" src="https://github.com/constantinosmch/Decision-Support-System-using-Model-Interpretability/blob/main/CNN_Explainer/Figure_6.png" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">caption</span>
</span> -->

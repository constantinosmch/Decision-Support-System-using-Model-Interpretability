#Imports necessary libraries
import pandas as pd
import matplotlib.pyplot
import numpy as np
import warnings
import pickle
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rule_extraction import rule_extract,draw_tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
warnings.filterwarnings('ignore')

#reading the dataset
df = pd.read_csv('./Data/Symptom_dataset/Symptom_dataset.csv')

#looking how much percent each diseases having
df['prognosis'].value_counts(normalize = True)

# Declares the static categories from the dataset
CATEGORIES= ["Fungal infection","Allergy","GERD","Chronic cholestasis","Drug Reaction","Peptic ulcer disease","AIDS","Diabetes ","Gastroenteritis","Bronchial Asthma","Hypertension ",
"Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)","Jaundice","Malaria","Chicken pox",
"Dengue","Typhoid","hepatitis A","Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E","Alcoholic hepatitis",
"Tuberculosis","Common Cold","Pneumonia","Dimorphic hemmorhoids(piles)","Heart attack","Varicose veins","Hypothyroidism",
"Hyperthyroidism","Hypoglycemia","Osteoarthristis","Arthritis","(vertigo) Paroymsal Positional Vertigo","Acne","Urinary tract infection",
"Psoriasis","Impetigo"]

# Seperated the independent and dependent values to repective variables 
x = df.drop(['prognosis'],axis =1)
y = df['prognosis']

# Divided into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialises decision tree classifier
dt = DecisionTreeClassifier(criterion="entropy", max_depth=9)
# Initialises random forest classifier
ran = RandomForestClassifier(n_estimators = 25, max_depth=5)
# Trains models
algo_model = dt.fit(x_train,y_train)
ran_model = ran.fit(x_train,y_train)

# Defines user input 
a = list(range(2,134))
i_name  = (input('Enter your name :'))
i_age = (int(input('Enter your age:')))
for i in range(len(x.columns)):
     print(str(i+1+1) + ":", x.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(x) for x in choices.split()]
count = 0
while count < len(b):
    item_to_replace =  b[count]
    replacement_value = 1
    indices_to_replace = [i for i,x in enumerate(a) if x==item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
a = [0 if x !=1 else x for x in a]

# Generated prediction and confidence
y_diagnosis = dt.predict([a])
y_pred_2 = dt.predict_proba([a])

print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
print(('Name = %s , Age : = %s') %(i_name,i_age))

# Generates random forest explanation
draw_tree(model = ran_model, outdir='./DecisionTree_RandomForest_Diagrams',feature_names = x_train.columns, proportion = False, class_names= [CATEGORIES[0],CATEGORIES[1] ,CATEGORIES[2] ,CATEGORIES[3],CATEGORIES[4] ,CATEGORIES[5] ,CATEGORIES[6],CATEGORIES[7] ,CATEGORIES[8] ,CATEGORIES[9],CATEGORIES[10] ,CATEGORIES[11], CATEGORIES[12],	CATEGORIES[13],	CATEGORIES[14],	CATEGORIES[15],	CATEGORIES[16],	CATEGORIES[17],	CATEGORIES[18],	CATEGORIES[19],	CATEGORIES[20],	CATEGORIES[21],	CATEGORIES[22],	CATEGORIES[23],	CATEGORIES[24],	CATEGORIES[25],	CATEGORIES[26],	CATEGORIES[27],	CATEGORIES[28],	CATEGORIES[29],	CATEGORIES[30],	CATEGORIES[31],	CATEGORIES[32],	CATEGORIES[33],	CATEGORIES[34],	CATEGORIES[35],	CATEGORIES[36],	CATEGORIES[37],	CATEGORIES[38],	CATEGORIES[39],CATEGORIES[40]])
# Saves trained models
filename = './models/DT.sav'
pickle.dump(algo_model, open(filename, 'wb'))
filename1 = './models/RF.sav'
pickle.dump(ran_model, open(filename1, 'wb'))

# Generates prediction
pred= algo_model.predict(x_test)

# Uncomment this section to view confusion matrices, recall and precision
# Generates confusion matrix
# cm=confusion_matrix(y_test,pred)
# print(cm)

# df_cm = pd.DataFrame(cm, range(len(CATEGORIES)), range(len(CATEGORIES)))
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

# plt.show()

# print(precision_recall_fscore_support(y_test,pred, average='macro'))
# print(accuracy_score(y_test, pred, normalize=False)/len(pred))
# pred= ran_model.predict(x_test)

# cm=confusion_matrix(y_test,pred)
# print(cm)

# df_cm = pd.DataFrame(cm, range(len(CATEGORIES)), range(len(CATEGORIES)))

# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

# plt.show()

# print(precision_recall_fscore_support(y_test,pred, average='macro'))
# print(accuracy_score(y_test, pred, normalize=False)/len(pred))

# Generates DT explanation
dot_data = StringIO()
export_graphviz(algo_model, 
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True,
                feature_names=x_train.columns,
                class_names=[CATEGORIES[0],CATEGORIES[1] ,CATEGORIES[2] ,CATEGORIES[3],CATEGORIES[4] ,CATEGORIES[5] ,CATEGORIES[6],CATEGORIES[7] ,CATEGORIES[8] ,CATEGORIES[9],CATEGORIES[10] ,CATEGORIES[11], CATEGORIES[12],	CATEGORIES[13],	CATEGORIES[14],	CATEGORIES[15],	CATEGORIES[16],	CATEGORIES[17],	CATEGORIES[18],	CATEGORIES[19],	CATEGORIES[20],	CATEGORIES[21],	CATEGORIES[22],	CATEGORIES[23],	CATEGORIES[24],	CATEGORIES[25],	CATEGORIES[26],	CATEGORIES[27],	CATEGORIES[28],	CATEGORIES[29],	CATEGORIES[30],	CATEGORIES[31],	CATEGORIES[32],	CATEGORIES[33],	CATEGORIES[34],	CATEGORIES[35],	CATEGORIES[36],	CATEGORIES[37],	CATEGORIES[38],	CATEGORIES[39],'40'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('./DecisionTree_RandomForest_Diagrams/DecisionTree.png')
Image(graph.create_png(), width=1920, height=640)
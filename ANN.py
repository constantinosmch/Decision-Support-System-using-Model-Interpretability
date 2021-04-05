#imported important libraries
import pandas as pd
import matplotlib.pyplot
import numpy as np
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
import pickle
import lime
import lime.lime_tabular
import webbrowser
warnings.filterwarnings('ignore')

# Declaring the static categories from the dataset
CATEGORIES= ["Fungal infection","Allergy","GERD","Chronic cholestasis",
"Drug Reaction","Peptic ulcer diseae","AIDS","Diabetes","Gastroenteritis",
"Bronchial Asthma","Hypertension","Migraine","Cervical spondylosis",
"Paralysis (brain hemorrhage)","Jaundice","Malaria","Chicken pox","Dengue",
"Typhoid","hepatitis A","Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E",
"Alcoholic hepatitis","Tuberculosis","Common Cold","Pneumonia","Dimorphic hemmorhoids(piles)",
"Heart attack","Varicose veins","Hypothyroidism","Hyperthyroidism","Hypoglycemia",
"Osteoarthristis","Arthritis","(vertigo) Paroymsal  Positional Vertigo","Acne",
"Urinary tract infection","Psoriasis","Impetigo"]

# Declaring the static symptoms from the dataset
SYMPTOMS = [ "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing",
"shivering","chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue",
"muscle_wasting","vomiting","burning_micturition","spotting_ urination","fatigue",
"weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss",
"restlessness","lethargy","patches_in_throat","irregular_sugar_level","cough",
"high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion",
"headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
"back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
"yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
"swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation",
"redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs",
"fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool",
"irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs",
"swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails",
"swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips",
"slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints",
"movement_stiffness","spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side",
"loss_of_smell","bladder_discomfort","foul_smell_of urine","continuous_feel_of_urine","passage_of_gases",
"internal_itching","toxic_look_(typhos)","depression","irritability","muscle_pain","altered_sensorium",
"red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes",
"increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration",
"visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma",
"stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload",
"blood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples",
"blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails",
"blister","red_sore_around_nose","yellow_crust_ooze"]

# Reads the dataset
df = pd.read_csv('./Data/Symptom_dataset/Symptom_dataset.csv')
 
df['prognosis'].value_counts(normalize = True)

# Seperates the independent and dependent values to repective variables 
x = df.drop(['prognosis'],axis =1)
y = df['prognosis']
tmpy =[]

# Label encoding for prognosis
for prognosis in y:
    if(prognosis == "Fungal infection"):
        tmpy.append(0)
    elif(prognosis == "Allergy"):
        tmpy.append(1)
    elif(prognosis == "GERD"):
        tmpy.append(2)
    elif(prognosis == "Chronic cholestasis"):
        tmpy.append(3)
    elif(prognosis == "Drug Reaction"):
        tmpy.append(4)
    elif(prognosis == "Peptic ulcer diseae"):
        tmpy.append(5)
    elif(prognosis == "AIDS"):
        tmpy.append(6)
    elif(prognosis == "Diabetes "):
        tmpy.append(7)
    elif(prognosis == "Gastroenteritis"):
        tmpy.append(8)
    elif(prognosis == "Bronchial Asthma"):
        tmpy.append(9)
    elif(prognosis == "Hypertension "):
        tmpy.append(10)
    elif(prognosis == "Migraine"):
        tmpy.append(11)
    elif(prognosis == "Cervical spondylosis"):
        tmpy.append(12)
    elif(prognosis == "Paralysis (brain hemorrhage)"):
        tmpy.append(13)
    elif(prognosis == "Jaundice"):
        tmpy.append(14)
    elif(prognosis == "Malaria"):
        tmpy.append(15)
    elif(prognosis == "Chicken pox"):
        tmpy.append(16)
    elif(prognosis == "Dengue"):
        tmpy.append(17)
    elif(prognosis == "Typhoid"):
        tmpy.append(18)
    elif(prognosis == "hepatitis A"):
        tmpy.append(19)
    elif(prognosis == "Hepatitis B"):
        tmpy.append(20)
    elif(prognosis == "Hepatitis C"):
        tmpy.append(21)
    elif(prognosis == "Hepatitis D"):
        tmpy.append(22)
    elif(prognosis == "Hepatitis E"):
        tmpy.append(23)
    elif(prognosis == "Alcoholic hepatitis"):
        tmpy.append(24)
    elif(prognosis == "Tuberculosis"):
        tmpy.append(25)
    elif(prognosis == "Common Cold"):
        tmpy.append(26)
    elif(prognosis == "Pneumonia"):
        tmpy.append(27)
    elif(prognosis == "Dimorphic hemmorhoids(piles)"):
        tmpy.append(28)
    elif(prognosis == "Heart attack"):
        tmpy.append(29)
    elif(prognosis == "Varicose veins"):
        tmpy.append(30)
    elif(prognosis == "Hypothyroidism"):
        tmpy.append(31)
    elif(prognosis == "Hyperthyroidism"):
        tmpy.append(32)
    elif(prognosis == "Hypoglycemia"):
        tmpy.append(33)
    elif(prognosis == "Osteoarthristis"):
        tmpy.append(34)
    elif(prognosis == "Arthritis"):
        tmpy.append(35)
    elif(prognosis == "(vertigo) Paroymsal  Positional Vertigo"):
        tmpy.append(36)
    elif(prognosis == "Acne"):
        tmpy.append(37)
    elif(prognosis == "Urinary tract infection"):
        tmpy.append(38)
    elif(prognosis == "Psoriasis"):
        tmpy.append(39)
    elif(prognosis == "Impetigo"):
        tmpy.append(40)
    else:
        print(prognosis)
        tmpy.append(41)

y= tmpy

#divided into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=53)

# One hot encode y labels 
y_train = to_categorical(y_train)
y_test =  to_categorical(y_test)

# Uncomment to re-train model
# # Declares Model
# model = Sequential()
# # Adds Dense layers
# model.add(Dense(y_train.shape[1],input_dim = x_train.shape[1]))
# model.add(Dense(y_train.shape[1],activation='softmax'))
# model.add(Dense(y_train.shape[1], activation='relu'))
# model.add(Dense(y_train.shape[1], activation='softmax'))
# # Compiles model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # Trains model
# model.fit(x_train, y_train, epochs=2 , batch_size= 5)
# filename = './models/ANN.sav'
# # Saves model
# pickle.dump(model, open(filename, 'wb'))

# Input symptoms: itching skin_rash nodal_skin_eruptions dischromic_patches
# When compiling the python application in order to view fungal_infection example
y1=[]
# Loads Model
filename = './models/ANN.sav'
model = pickle.load(open(filename, 'rb'))
print("Please enter the symptoms:")
# Declares user input
input1 = input() 
text= input1.split()
# Appends temp array
for i in SYMPTOMS: 
    y1.append(0)
# Replaces symptoms which are experienced with a value of 1 in the temp array
for i in range(len(text)):
    y1[SYMPTOMS.index(text[i])] = 1
y1= np.array(y1)  
kala = pd.DataFrame([y1],columns =SYMPTOMS)
# Generates raw prediction of the instance using the temp array
prediction = model.predict(kala)
# Choses the most confindent prediction
most_conf_index = np.argmax(prediction)
# Provides the percentage of certainty
answer_confidence = prediction[0][most_conf_index]
 
print("Model classified prognosis as", CATEGORIES[most_conf_index], "with", answer_confidence,"confidence")

# Analyses the symptoms with the same diagnosis from the dataset
for z in range(len(text)):
    count = 0
    count1 = 0
    prob = 0
    for i in range(len(x[text[z]])):    
        if(df['prognosis'][i] == CATEGORIES[most_conf_index]): 
            count1 = count1 + 1

    for i in range(len(x[text[z]])):    
        if(df[text[z]][i]==1 and df['prognosis'][i] == CATEGORIES[most_conf_index]): 
            count = count + 1
    prob = count/count1
    if(prob>=0.9):
        print("The symptom:", text[z] , "has more than 90% probability to appear when the prognosis is ", CATEGORIES[most_conf_index])
    elif(prob<=0.01):
        print("The symptom:", text[z] , "has less than 1% probability to appear when the prognosis is ", CATEGORIES[most_conf_index])
    else:
        print("The symptom:", text[z] , "has", prob*100,"% probability to appear when the prognosis is ", CATEGORIES[most_conf_index])

# Other symptoms that can appear which have the same diagnosis
arr = []
# Adds symptoms to array
for i in range(len(df['prognosis'])):
    if(df['prognosis'][i] == CATEGORIES[most_conf_index-1]):
        for y in range(len(SYMPTOMS)):
            if(df[SYMPTOMS[y]][i] == 1):
                arr.append(SYMPTOMS[y])
                arr = list(dict.fromkeys(arr))
                
temp = []
for i in range(len(text)):
    for j in range(len(arr)):
        if(text[i]==arr[j]):
            temp.append(arr[j])
# Removes duplicate values
for i in range(len(temp)):
    arr.remove(temp[i])
# Prints symptoms 
print("Other symptoms include:")
for x in range(len(arr)):
    print(arr[x])

# Loads model 
filename = './models/ANN.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(x_test)

count = 0 
for i in range(len(x_test)):
    prediction1 = model.predict(x_test[i:i+1])
    most_conf_index1 = np.argmax(prediction1)
    if(CATEGORIES[most_conf_index1] == result[i]):
            count = count +1

print(count/len(x_test))

# Converts x_train and x_test to np array
x_train = np.array(x_train)
x_test = np.array(x_test)
# Uses Lime explainer to generate explanation with the predifined symptoms and categories
explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=list(SYMPTOMS),class_names= CATEGORIES, discretize_continuous=True)
exp = explainer.explain_instance(y1, model.predict, num_features=len(SYMPTOMS), top_labels=len(CATEGORIES))
# Saves explanation as html file
file_path = 'ANN_Explainer/explain.html/.'
exp.save_to_file(file_path, labels=None, predict_proba=True, show_predicted_value=True)
# Opens saved html file
# To view file change URL accordingly
url = 'file:///C:/Users/Constantinos/Desktop/Diagnosis/ANN_Explainer/explain.html'
# opens html link in a new web browser tab
webbrowser.open(url, new=2)
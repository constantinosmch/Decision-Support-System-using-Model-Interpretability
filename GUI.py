# Import libraries
import tkinter as tk
from tkinter import filedialog
import pickle
import os
from keras.preprocessing import image
import numpy as np
from keras.applications import inception_v3 as inc_net
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import pandas as pd
from sklearn.model_selection import train_test_split  
from keras.utils import to_categorical
import lime.lime_tabular
rep1=[]
ar=[]
IMG_SIZE = 128

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
# Divided into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=53)
y_train = to_categorical(y_train)
y_test =  to_categorical(y_test)

# Initialises GUI Class
class GUI():
    # Main Menu layout  
    def __init__(self):
        self.root = tk.Tk()
        self.text = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.text)
        self.root.geometry("500x300+120+120")
        self.label = tk.Label(self.root,text = "Main Menu")
        self.label.grid(row=0,column=1)
        self.button = tk.Button(self.root,text = "ANN Symptom Checker",command=self.open_window)
        self.button.grid(row=1,column=0)
        self.button1 = tk.Button(self.root,text="MRI", command=self.open_MRI_window)
        self.button1.grid(row=1,column=1)
        self.button2 = tk.Button(self.root,text="RF Symptom Checker", command=self.open_windowRF)
        self.button2.grid(row=1,column=2)
        self.button3 = tk.Button(self.root,text="Quit", command=self.root.destroy)
        self.button3.grid(row=1,column=3)
        self.root.mainloop()
    
    # Generates CNN prediction using pre-trained model  
    def mripred(self):
        def transform_img_fn(path_list):
            out = []
            for img_path in path_list:
                img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = inc_net.preprocess_input(x)
                out.append(x)
            return np.vstack(out)  
        CATEGORIES1 = ['No tumour was deteced','A tumour was deteced']
        filename = './models/CNN.sav'
        inet_model = pickle.load(open(filename, 'rb'))
        st=""

        for i in range(len(rep1)-1):
            st = st+rep1[i]+"/"
        st1 = rep1[len(rep1)-1]
        images = transform_img_fn([os.path.join(st,st1)])
        prediction = inet_model.predict(images)[0]
        print("Raw prediction made by model:", prediction)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[most_conf_index]
        print("Model classified image as", CATEGORIES1[most_conf_index], "with", answer_confidence,"confidence")
        self.text.set(CATEGORIES1[most_conf_index]) 
        rep1.clear()
    
    # Generates CNN prediction and explanation using pre-trained model and Lime    
    def mripredexplain(self):
        def transform_img_fn(path_list):
            out = []
            for img_path in path_list:
                img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = inc_net.preprocess_input(x)
                out.append(x)
            return np.vstack(out)  
        CATEGORIES1 = ['No tumour was deteced','A tumour was deteced']
        filename = './models/CNN.sav'
        inet_model = pickle.load(open(filename, 'rb'))
        st=""
        print(rep1)
        for i in range(len(rep1)-1):
            st = st+rep1[i]+"/"
        print(st)
        st1 = rep1[len(rep1)-1]
        print(rep1[len(rep1)-1])

        images = transform_img_fn([os.path.join(st,st1)])
        prediction = inet_model.predict(images)[0]
        print("Raw prediction made by model:", prediction)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[most_conf_index]
        print("Model classified image as", CATEGORIES1[most_conf_index], "with", answer_confidence,"confidence")
        self.text.set(CATEGORIES1[most_conf_index])
        images = images.reshape(IMG_SIZE,IMG_SIZE,3)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images, inet_model.predict, top_labels=5, hide_color=1, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
        rep1.clear()

    # Opens MRI window menu
    def open_MRI_window(self):
        self.top = tk.Toplevel()
        self.top.title("MRI Diagnosis")
        self.top.geometry("500x300+120+120")
        self.button = tk.Button(self.top,text = "Back",command=self.top.destroy)
        self.button.grid(row =3,column=2)
        self.label_1 = tk.Label(self.top,text="MRI")
        self.label_1.grid(row = 1,column=1)
        self.button1 = tk.Button(self.top,text = "Open files",command=self.c_open_file_old)
        self.button1.grid(row =2)
        self.button2 = tk.Button(self.top,text = "Go",command=self.mripred)
        self.button2.grid(row=3,column=1)  
        self.button3 = tk.Button(self.top,text = "Show Explanation",command=self.mripredexplain)
        self.button3.grid(row=3,column=0)  
        self.text = tk.StringVar()
        self.text.set("No prediction made")
        self.label_3 = tk.Label(self.top,text="Prediction:")
        self.label_3.grid(row =2, column= 2)
        self.label_2 = tk.Label(self.top,textvariable=self.text)
        self.label_2.grid(row =2, column= 3)

    
    # Refreshes MRI menu 
    def c_open_file_old(self):
        self.top = tk.Toplevel()
        self.top.title("MRI Diagnosis")
        self.top.geometry("500x300+120+120")
        self.button = tk.Button(self.top,text = "Back",command=self.top.destroy)
        self.button.grid(row=3,column=2)
        self.label_1 = tk.Label(self.top,text="MRI")
        self.label_1.grid(row = 1,column=1)
        self.button1 = tk.Button(self.top,text = "Open files")
        self.button1.grid(row =2)
        self.button2 = tk.Button(self.top,text = "Go")
        self.button2.grid(row=3,column=1)
        self.button3 = tk.Button(self.top,text = "Show Explanation")
        self.button3.grid(row=3,column=0)  
        self.label_3 = tk.Label(self.top,text="Prediction:")
        self.label_3.grid(row =2, column= 2)
        self.label_2 = tk.Label(self.top,textvariable=self.text)
        self.label_2.grid(row =2, column= 3)

        rep = tk.filedialog.askopenfilenames(
            parent=self.top,
            initialdir='/',
            initialfile='tmp',
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*")])
        rep2 = rep[0].split("/")
        try:
            for i in range(len(rep2)):
                rep1.append(rep2[i])
            print(rep[0])
        except IndexError:
            print("No file selected")    
        self.top.destroy()

    # Adds symptom to existing array    
    def add(self):
        ar.append(self.variable.get())
        self.label_2['text']="\n".join(ar)

    # Clears symptom array
    def cleararr(self):
        ar.clear()
        self.text.set("No prediction made")
        self.label_2['text']=""

    # Generates an ANN explanation using the pre-trained model
    def predwithoutexplain(self):
        filename = './models/ANN.sav'
        model = pickle.load(open(filename, 'rb'))
        y1=[]
        text= ar
        for i in SYMPTOMS: 
            y1.append(0)
        for i in range(len(text)):
            y1[SYMPTOMS.index(text[i])] = 1
        y1= np.array(y1)  
        kala = pd.DataFrame([y1],columns =SYMPTOMS)
        prediction = model.predict(kala)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[0][most_conf_index]
        self.text.set(CATEGORIES[most_conf_index])
        print("Model classified prognosis as", CATEGORIES[most_conf_index], "with", answer_confidence,"confidence")

    # Generates a ANN prediction and explanation using the pre-trained model and Lime
    def pred(self):
        filename = './models/ANN.sav'
        model = pickle.load(open(filename, 'rb'))
        y1=[]
        text= ar
        for i in SYMPTOMS: 
            y1.append(0)
        for i in range(len(text)):
            y1[SYMPTOMS.index(text[i])] = 1
        y1= np.array(y1)  
        kala = pd.DataFrame([y1],columns =SYMPTOMS)
        prediction = model.predict(kala)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[0][most_conf_index]
        x_train1 = np.array(x_train)
        x_test1 = np.array(x_test)
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train1, feature_names=list(SYMPTOMS),class_names= CATEGORIES, discretize_continuous=True)
        i = np.random.randint(0, x_test1.shape[0])
        exp = explainer.explain_instance(y1, model.predict, num_features=len(SYMPTOMS), top_labels=len(CATEGORIES))
        file_path = 'ANN_Explainer/explain.html/.'
        exp.save_to_file(file_path, labels=None, predict_proba=True, show_predicted_value=True)
        import webbrowser
        url = 'file:///C:/Users/Constantinos/Desktop/Diagnosis/ANN_Explainer/explain.html'
        webbrowser.open(url, new=2)  # open in new tab
        self.text.set(CATEGORIES[most_conf_index])
        print("Model classified prognosis as", CATEGORIES[most_conf_index], "with", answer_confidence,"confidence")
    
    # Opens ANN window menu
    def open_window(self):
        self.top = tk.Toplevel()
        self.top.title("Symptom Diagnosis")
        self.top.geometry("500x300+120+120")
        self.button = tk.Button(self.top,text = "Back",command=self.top.destroy)
        self.button.grid(row=3,column=2)
        self.label_1 = tk.Label(self.top,text="Symptoms")
        self.label_1.grid(row = 1)
        self.button1 = tk.Button(self.top,text = "Show Explanation",command=self.pred)
        self.button1.grid(row=3,column=0)
        self.button1 = tk.Button(self.top,text = "Go",command=self.predwithoutexplain)
        self.button1.grid(row=3,column=1)
        self.variable = tk.StringVar(self.top)
        self.variable.set("Select Symptoms") # default value
        self.w = tk.OptionMenu(self.top, self.variable, *sorted(SYMPTOMS))
        self.w.grid(row=1,column=1)
        self.button2 = tk.Button(self.top,text = "Add",command=self.add)
        self.button2.grid(row=1,column=3)
        self.button3 = tk.Button(self.top,text = "Clear",command=self.cleararr)
        self.button3.grid(row=2,column=3)
        self.label_2 = tk.Label(self.top,text="")
        self.label_2.grid(row=2)
        self.label_3 = tk.Label(self.top,text="Prediction:")
        self.label_3.grid(row =2, column= 4)
        self.text = tk.StringVar()
        self.text.set("No prediction made")
        self.label_4 = tk.Label(self.top,textvariable=self.text)
        self.label_4.grid(row =2, column= 5)

    # Generates a prediction using the pre-trained RF model
    def predwithoutexplainRF(self):
        filename = './models/RF.sav'
        model = pickle.load(open(filename, 'rb'))
        y1=[]
        text= ar
        for i in SYMPTOMS: 
            y1.append(0)
        for i in range(len(text)):
            y1[SYMPTOMS.index(text[i])] = 1
        y1= np.array(y1)  
        kala = pd.DataFrame([y1],columns =SYMPTOMS)
        prediction = model.predict(kala)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[0][most_conf_index]
        self.text.set(CATEGORIES[most_conf_index])
        print("Model classified prognosis as", CATEGORIES[most_conf_index])
    
    # Grenerates a prediction and explanation using the pre-trained RF model
    def predRF(self):
        filename = './models/RF.sav'
        model = pickle.load(open(filename, 'rb'))
        y1=[]
        text= ar
        for i in SYMPTOMS: 
            y1.append(0)
        for i in range(len(text)):
            y1[SYMPTOMS.index(text[i])] = 1
        y1= np.array(y1)  
        kala = pd.DataFrame([y1],columns =SYMPTOMS)
        prediction = model.predict(kala)
        most_conf_index = np.argmax(prediction)
        answer_confidence = prediction[0][most_conf_index]
        self.text.set(CATEGORIES[most_conf_index])
        os.startfile('C:/Users/Constantinos/Desktop/Diagnosis/DecisionTree_RandomForest_Diagrams/DecisionTree.png')
        print("Model classified prognosis as", CATEGORIES[most_conf_index])

    # Opens RF menu
    def open_windowRF(self):
        self.top = tk.Toplevel()
        self.top.title("Symptom Diagnosis")
        self.top.geometry("500x300+120+120")
        self.button = tk.Button(self.top,text = "Back",command=self.top.destroy)
        self.button.grid(row=3,column=2)
        self.label_1 = tk.Label(self.top,text="Symptoms")
        self.label_1.grid(row = 1)
        self.button1 = tk.Button(self.top,text = "Show Explanation",command=self.predRF)
        self.button1.grid(row=3,column=0)
        self.button1 = tk.Button(self.top,text = "Go",command=self.predwithoutexplainRF)
        self.button1.grid(row=3,column=1)
        self.variable = tk.StringVar(self.top)
        self.variable.set("Select Symptoms") # default value
        self.w = tk.OptionMenu(self.top, self.variable, *sorted(SYMPTOMS))
        self.w.grid(row=1,column=1)
        self.button2 = tk.Button(self.top,text = "Add",command=self.add)
        self.button2.grid(row=1,column=3)
        self.button3 = tk.Button(self.top,text = "Clear",command=self.cleararr)
        self.button3.grid(row=2,column=3)
        self.label_2 = tk.Label(self.top,text="")
        self.label_2.grid(row=2)
        self.label_3 = tk.Label(self.top,text="Prediction:")
        self.label_3.grid(row =2, column= 4)
        self.text = tk.StringVar()
        self.text.set("No prediction made")
        self.label_4 = tk.Label(self.top,textvariable=self.text)
        self.label_4.grid(row =2, column= 5)

app=GUI()
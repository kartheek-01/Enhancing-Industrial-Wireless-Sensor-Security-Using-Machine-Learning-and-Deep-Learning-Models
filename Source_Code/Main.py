import os 
# Suppress TensorFlow GPU warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tkinter 
from tkinter import * 
from tkinter import filedialog 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential 
from keras.layers import Dense 

# --- Main Window Setup ---
main = tkinter.Tk() 
main.title("Network Intrusion Detection") 
main.geometry("1300x750") 
main.config(bg='PeachPuff2') 

# --- Global Variables ---
filename = "" 
X = Y = X_train = X_test = y_train = y_test = None 
svm_acc = ann_acc = LR_acc = DT_acc = RFT_acc = 0 
classifier = None 
selector_svm = None 

def isfloat(value): 
    try: 
        float(value) 
        return True 
    except ValueError: 
        return False 

def splitdataset(balance_data): 
    # Use relative indexing to automatically adjust to feature vs label boundaries
    X_raw = balance_data.iloc[:, :-1].values 
    Y_raw = balance_data.iloc[:, -1].values 
    
    scaler = MinMaxScaler() 
    X_scaled = scaler.fit_transform(X_raw) 
    
    X_train, X_test, y_train, y_test = train_test_split( 
        X_scaled, Y_raw, test_size=0.2, random_state=0 
    ) 
    return X_scaled, Y_raw, X_train, X_test, y_train, y_test 

def upload(): 
    global filename 
    text.delete('1.0', END) 
    filename = filedialog.askopenfilename(initialdir=".") 
    if filename:
        pathlabel.config(text=filename) 
        text.insert(END, "Dataset Loaded Successfully\n\n") 
    else:
        pathlabel.config(text="No file selected")

def preprocess(): 
    global filename 
    text.delete('1.0', END) 
    if not filename: 
        text.insert(END, "Please upload a dataset first.\n") 
        return 
    try: 
        df = pd.read_csv(filename) 
        text.insert(END, "Removed non-numeric characters from dataset and saved inside clean.txt file\n\n") 
        text.insert(END, "Dataset Information\n\n") 
        text.update() 
        
        # Process first 41 feature columns to numeric, fill NaN with 0
        processed_df = df.iloc[:, 0:41].apply(pd.to_numeric, errors='coerce').fillna(0) 
        
        # Binary label mapping (normal vs anomaly)
        labels = df.iloc[:, 41].astype(str).str.lower().apply(lambda x: 0 if 'normal' in x else 1) 
        processed_df['label'] = labels 
        
        # Save to file 
        processed_df.to_csv("clean.txt", index=False, header=False) 
        
        # Show the processed row samples (Top 15 rows) 
        sample_data = processed_df.head(15).astype(str).values 
        for row in sample_data: 
            line = ",".join(row) + "\n" 
            text.insert(END, line) 
        text.update() 
    except Exception as e: 
        text.insert(END, f"Error during preprocessing: {e}\n") 

def generateModel(): 
    global X, Y, X_train, X_test, y_train, y_test 
    text.delete('1.0', END) 
    if not os.path.exists("clean.txt"): 
        text.insert(END, "Cleaned file not found. Please preprocess first.\n") 
        return 
    try: 
        balance_data = pd.read_csv("clean.txt", header=None) 
        X, Y, X_train, X_test, y_train, y_test = splitdataset(balance_data) 
        text.insert(END, "Train & Test Model Generated\n\n") 
        text.insert(END, f"Total Dataset Size : {len(balance_data)}\n") 
        text.insert(END, f"Training Size : {len(X_train)}\n") 
        text.insert(END, f"Testing Size : {len(X_test)}\n") 
    except Exception as e: 
        text.insert(END, f"Error: {e}\n") 

def runSVM(): 
    global svm_acc, classifier, selector_svm 
    text.delete('1.0', END) 
    if X_train is None: 
        text.insert(END, "Generate Training Model first.\n") 
        return 
    
    # Chi-Square features must be non-negative (MinMax Scaler guarantees this)
    selector_svm = SelectKBest(score_func=chi2, k=15) 
    X_train1 = selector_svm.fit_transform(X_train, y_train.astype('int')) 
    X_test1 = selector_svm.transform(X_test) 
    
    model = svm.SVC(kernel='rbf', probability=True) 
    model.fit(X_train1, y_train.astype('int')) 
    pred = model.predict(X_test1) 
    
    svm_acc = accuracy_score(y_test.astype('int'), pred) * 100 
    classifier = model 
    text.insert(END, f"SVM Accuracy : {svm_acc:.2f}%\n\n") 

def runDT(): 
    global DT_acc 
    text.delete('1.0', END) 
    if X_train is None: 
        text.insert(END, "Generate Training Model first.\n") 
        return 
    
    selector = SelectKBest(score_func=chi2, k=15) 
    X_train1 = selector.fit_transform(X_train, y_train.astype('int')) 
    X_test1 = selector.transform(X_test) 
    
    model = DecisionTreeClassifier() 
    model.fit(X_train1, y_train.astype('int')) 
    pred = model.predict(X_test1) 
    
    DT_acc = accuracy_score(y_test.astype('int'), pred) * 100 
    text.insert(END, f"Decision Tree Accuracy : {DT_acc:.2f}%\n\n") 

def runRFT(): 
    global RFT_acc 
    text.delete('1.0', END) 
    if X_train is None: 
        text.insert(END, "Generate Training Model first.\n") 
        return 
    
    selector = SelectKBest(score_func=chi2, k=15) 
    X_train1 = selector.fit_transform(X_train, y_train.astype('int')) 
    X_test1 = selector.transform(X_test) 
    
    model = RandomForestClassifier() 
    model.fit(X_train1, y_train.astype('int')) 
    pred = model.predict(X_test1) 
    
    RFT_acc = accuracy_score(y_test.astype('int'), pred) * 100 
    text.insert(END, f"Random Forest Accuracy : {RFT_acc:.2f}%\n\n") 

def runLR(): 
    global LR_acc 
    text.delete('1.0', END) 
    if X_train is None: 
        text.insert(END, "Generate Training Model first.\n") 
        return 
    
    selector = SelectKBest(score_func=chi2, k=15) 
    X_train1 = selector.fit_transform(X_train, y_train.astype('int')) 
    X_test1 = selector.transform(X_test) 
    
    model = LogisticRegression(max_iter=500) 
    model.fit(X_train1, y_train.astype('int')) 
    pred = model.predict(X_test1) 
    
    LR_acc = accuracy_score(y_test.astype('int'), pred) * 100 
    text.insert(END, f"Logistic Regression Accuracy : {LR_acc:.2f}%\n\n") 

def runANN(): 
    global ann_acc 
    text.delete('1.0', END) 
    if X_train is None: 
        text.insert(END, "Generate Training Model first.\n") 
        return 
    
    selector = SelectKBest(score_func=chi2, k=25) 
    X_train1 = selector.fit_transform(X_train, y_train.astype('int')) 
    X_test1 = selector.transform(X_test) 
    
    model = Sequential() 
    model.add(Dense(30, input_dim=25, activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit(X_train1, y_train.astype('int'), epochs=10, batch_size=32, verbose=0) 
    
    loss, accuracy = model.evaluate(X_test1, y_test.astype('int'), verbose=0) 
    ann_acc = accuracy * 100 
    text.insert(END, f"ANN Accuracy : {ann_acc:.2f}%\n\n") 

def detectAttack(): 
    global classifier, selector_svm 
    text.delete('1.0', END) 
    if classifier is None or selector_svm is None: 
        text.insert(END, "Please run SVM first to train the reference classifier and selector.\n") 
        return 
    
    file = filedialog.askopenfilename(initialdir=".") 
    if file: 
        try: 
            text.insert(END, "Test Data Uploaded\n\n") 
            text.update() 
            
            # Read test file safely handling potential header variations
            test_df = pd.read_csv(file) 
            if not isfloat(str(test_df.iloc[0, 0])): 
                pass 
            elif isfloat(str(test_df.columns[0])): 
                test_df = pd.read_csv(file, header=None) 
            
            # Keep features only (columns 0 to 41 matching the training layout)
            test_data = test_df.iloc[:, 0:41].apply(pd.to_numeric, errors='coerce').fillna(0).values 
            
            scaler = MinMaxScaler() 
            test_data_scaled = scaler.fit_transform(test_data) 
            test_selected = selector_svm.transform(test_data_scaled) 
            
            y_pred = classifier.predict(test_selected) 
            for i, res in enumerate(y_pred): 
                msg = f"Row {i+1}: Detected Attack\n" if res == 1 else f"Row {i+1}: Normal Traffic\n" 
                text.insert(END, msg) 
        except Exception as e: 
            text.insert(END, f"Error during detection: {e}\n") 

def graph(): 
    height = [svm_acc, ann_acc, LR_acc, DT_acc, RFT_acc] 
    bars = ('SVM', 'ANN', 'LR', 'DT', 'RFT') 
    y_pos = np.arange(len(bars)) 
    
    plt.figure(figsize=(10, 6)) 
    plt.bar(y_pos, height, color=['blue', 'orange', 'green', 'red', 'purple']) 
    plt.xticks(y_pos, bars) 
    plt.ylabel("Accuracy %") 
    plt.title("Algorithm Accuracy Comparison") 
    plt.ylim(0, 110) 
    plt.show() 

# --- UI Setup --- 
font = ('times', 16, 'bold') 
title = Label(main, text='Network Intrusion Detection using Machine Learning', 
              bg='PaleGreen2', fg='Khaki4', font=font, height=3, width=120) 
title.place(x=0, y=5) 

pathlabel = Label(main, text="No file selected", bg='PeachPuff2') 
pathlabel.place(x=700, y=150) 

Button(main, text="Upload NSL KDD Dataset", command=upload, width=35).place(x=700, y=100) 
Button(main, text="Preprocess Dataset", command=preprocess, width=35).place(x=700, y=200) 
Button(main, text="Generate Training Model", command=generateModel, width=35).place(x=700, y=250) 
Button(main, text="Run SVM Algorithm", command=runSVM, width=35).place(x=700, y=300) 
Button(main, text="Run ANN Algorithm", command=runANN, width=35).place(x=700, y=350) 
Button(main, text="Run LR Algorithm", command=runLR, width=35).place(x=700, y=400) 
Button(main, text="Run DT Algorithm", command=runDT, width=35).place(x=700, y=450) 
Button(main, text="Run RFT Algorithm", command=runRFT, width=35).place(x=700, y=500) 
Button(main, text="Upload Test Data & Detect Attack", command=detectAttack, width=35).place(x=700, y=550) 
Button(main, text="Accuracy Graph", command=graph, width=35).place(x=700, y=600) 

text = Text(main, height=30, width=80, font=('times', 12, 'bold')) 
text.place(x=10, y=100) 

main.mainloop()

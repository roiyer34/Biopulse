# Importing libraries
from flask import Flask, flash, redirect, render_template, request, session, jsonify
import requests
from flask_session import Session
from tempfile import mkdtemp
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
from sqlite3 import Error
import os
from werkzeug.security import check_password_hash, generate_password_hash
import xml.etree.ElementTree as ET
import re
import json
from datetime import date

# Create sql database
db = os.path.realpath('Users.db')

# Connect to sql data
conn = None
try:
    conn = sqlite3.connect(db, check_same_thread=False)
except Error as e:
    print(e)

with open('schema.sql') as f:
    conn.executescript(f.read())

cur = conn.cursor()

# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Get dataset 
df = pd.read_csv("Dataset/Training.csv")

# Column of disease names. We will convert the prognosis column to a numeric
# Later on, so this will help us get back to strings
disease_names = df['prognosis']
df.drop("Unnamed: 133", axis='columns',inplace=True)

# Turn prognosis column into numeric for logistical regression
encoder = LabelEncoder()
df["prognosis"].replace(encoder.fit_transform(df["prognosis"]),inplace=True)
progs = list(df['prognosis'].unique())
# Make into a numeric type
replaceStruct = {
    'prognosis': {}
}
for val in  progs:
  replaceStruct['prognosis'][val] = progs.index(val)
df = df.replace(replaceStruct)



# Get x and y for logistical regression
x = df.drop('prognosis',axis='columns')
y=df['prognosis']


# Split the data into train and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)

# Logistic regression model with an example test case
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
logmodel=LogisticRegression(max_iter=1000,C=1)
logmodel.fit(x_train,y_train)

# MedlinePlus API base URL
MEDLINEPLUS_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

def clean_text(text):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Initialize an empty list to store cleaned sentences
    cleaned_sentences = []

    for sentence in sentences:
        # Remove leading and trailing whitespaces
        cleaned_sentence = sentence.strip()

        # Add a period at the end if it's missing
        if not cleaned_sentence.endswith('.'):
            cleaned_sentence += '.'

        # Add spaces after commas when listing symptoms
        cleaned_sentence = re.sub(r'([A-Za-z]),', r'\1, ', cleaned_sentence)

        # Add commas where needed (before "and" or "or")
        cleaned_sentence = re.sub(r' (and|or) ', r', \1 ', cleaned_sentence)

        # Add the cleaned sentence to the list
        cleaned_sentences.append(cleaned_sentence)

    # Join the cleaned sentences to form the cleaned text
    cleaned_text = ' '.join(cleaned_sentences)

    return cleaned_text

def extract_description(xml_content):
    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Initialize variables to store the most relevant description
    most_relevant_description = None
    highest_rank = float('inf')  # Initialize with positive infinity rank

    # Iterate through the search results and find the most relevant description
    for document in root.findall(".//document"):
        rank = int(document.get("rank", 0))

        # Check if the current document has a higher rank (lower value is better)
        if rank < highest_rank:
            highest_rank = rank
            description_element = document.find(".//content[@name='FullSummary']")

            if description_element is not None:
                most_relevant_description = description_element.text

    if most_relevant_description is not None:
        # Add spaces after periods and colons
        cleaned_description = re.sub(r'(?<=[.:])(?=[^\s])', ' ', most_relevant_description)

        return cleaned_description.strip()  # Return the cleaned description without leading/trailing spaces
    else:
        return None







def get_disease_info(disease_name):
    if not disease_name:
        return {"error": "Disease name not provided"}

    # Construct the query URL
    query_url = f"{MEDLINEPLUS_BASE_URL}?db=healthTopics&term={disease_name}"

    try:
        response = requests.get(query_url)
        response.raise_for_status()
        
        summary = extract_description(response.content)

        if summary:
            return summary
        else:
            return {"error": "Disease information not found"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to MedlinePlus API: {str(e)}"}
    

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        if "user_email" in session:
            return render_template("checklist.html", symptom_list = x.columns)
        return render_template("index.html", error = "")
    elif request.method == "POST":
        #user is trying to register
        if request.form.get('log_email')== None:
            #error checking
            if request.form.get("sign_email") == "" or  request.form.get("sign_password") == "":
                return render_template("index.html", error="Please input valid email and password")
            rows = cur.execute("SELECT * FROM users WHERE email = ?;", (request.form.get("sign_email"),)).fetchall()
            #more error checking
            if len(rows) >=1:
                return render_template("index.html", error="There is already an account with the given username")
            #insert user information into database
            if request.form.get("sign_confirm") == request.form.get("sign_password") and not request.form.get("sign_password") == "" and not request.form.get("sign_email") == "":
                cur.execute("INSERT INTO users (email, hash, prevConditions) VALUES (?,?, ?);", (request.form.get("sign_email"), generate_password_hash(request.form.get("sign_password")), json.dumps([])))
                conn.commit()
                return render_template("checklist.html", symptom_list = x.columns)
            else:
                return render_template("index.html", error="Passwords don't match")
            
        elif request.form.get('sign_email')== None:
            """Log user in"""


            # Forget any user_id
            session.clear()

            # User reached route via POST (as by submitting a form via POST)
            if request.method == "POST":

                # Ensure username was submitted
                if request.form.get("log_email") == "" or request.form.get("log_password") == "":
                    return render_template("index.html", error="Must enter email and password when logging in")

                # Query database for username
                rows = cur.execute("SELECT * FROM users WHERE email = ?", (request.form.get("log_email"),)).fetchall()
                # Ensure username exists and password is correct
                if len(rows) != 1 or not check_password_hash(rows[0][1], request.form.get("log_password")):
                    return render_template("index.html", error="Incorrect login information")
                # Remember which user has logged in
                session["user_email"] = rows[0][0]
                session["user_password"] = rows[0][1]

                # Redirect user to home page
                return render_template("checklist.html", symptom_list = x.columns)


        # if request.form.get()
        # #error checking
        # if not request.form.get("username") or not request.form.get("password"):
        #     return render_template("index.html", error = "Missing Username or Password")
        # rows = db.execute("SELECT * FROM users WHERE username = ?;", request.form.get("username"))
        # #more error checking
        # if len(rows) >=1:
        #     return render_template("index.html", error = "Username is taken")
        # #insert user information into database
        # if request.form.get("password") == request.form.get("confirmation") and request.form.get("username") and request.form.get("password"):
        #     db.execute("INSERT INTO users (username, hash) VALUES (?,?);", request.form.get("username"), generate_password_hash(request.form.get("password")))
        #     return render_template("login.html")
        # else:
        #     return apology("Passwords do not match")

@app.route("/checklist", methods = ['GET', 'POST'])
def checklist():
    if request.method == "GET":
        return render_template("checklist.html", symptom_list = x.columns)
    if request.method == "POST":
        if request.form.get("checklist"):
            return render_template("checklist.html", symptom_list = x.columns)
        elif request.form.get("result_log"):
            print(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            temp = json.loads(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            return render_template('resultlog.html', data=temp)
        elif request.form.get("user_profile"):
            return render_template("userprofile.html")
        elif request.form.get("logout"):
            session.clear()
            redirect("/")
        symptoms = list(map(int, request.form.getlist('symptom')))
        i = 0
        while i < len(symptoms) - 1:
            if symptoms[i] == 0 and symptoms[i+1] == 1:
                del symptoms[i]
            else:
                i += 1
        cpy = symptoms
        symptoms = [symptoms]
        string_symptoms = []
        for i in range(len(cpy)):
            if cpy[i] == 1:
                string_symptoms.append(df.columns[i])
        predictions = logmodel.predict(symptoms)
        diagnosis = progs[predictions[0]]
        # Get disease information
        description = get_disease_info(diagnosis)

        # Create the data dictionary
        data_to_render = {
            "condition": diagnosis,
            "selected_symptom_list": string_symptoms,  # Assuming symptoms is a list of symptoms
            "description": description  # Pass the description directly
        }
        temp = cur.execute("SELECT prevConditions FROM users WHERE email=(?)", session["user_email"]).fetchall()
        temp = temp[0][0]
        temp = json.loads(temp)
        temp.insert(0, [str(date.today()), diagnosis, string_symptoms])
        cur.execute("UPDATE users SET prevConditions=(?) WHERE email=(?);", (json.dumps(temp), session["user_email"]))
        # Pass the data to the template
        return render_template('results.html', **data_to_render)
    
@app.route("/results", methods = ['GET', 'POST'])
def results():
    if request.method == "GET":
        return render_template("checklist.html", symptom_list = x.columns)
    if request.method == "POST":
        if request.form.get("checklist"):
            return render_template("checklist.html")
        elif request.form.get("result_log"):
            print(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            temp = json.loads(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            return render_template('resultlog.html', data=temp)
        elif request.form.get("user_profile"):
            return render_template("userprofile.html")
        elif request.form.get("logout"):
            session.clear()
            redirect("/")

@app.route("/userprofile", methods = ['GET', 'POST'])
def userprofile():
    if request.method == "GET":
        return render_template('userprofile.html')
    if request.method == "POST":
        print(session)
        if request.form.get("checklist"):
            return render_template("checklist.html", symptom_list = x.columns)
        elif request.form.get("result_log"):
            print(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            temp = json.loads(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            return render_template('resultlog.html', data=temp)
        elif request.form.get("user_profile"):
            return render_template("userprofile.html")
        elif request.form.get("logout"):
            session.clear()
            redirect("/")
        #more error checking
        if(request.form.get("newUsername") != None):
            rows = cur.execute("SELECT * FROM users WHERE email = ?;", (request.form.get("newUsername"),)).fetchall()
            if(len(rows) == 0):
                cur.execute("UPDATE users SET email=(?) WHERE email=(?)", (request.form.get("newUsername"), session["user_email"]))
                conn.commit()
                session["user_email"] = request.form.get("newUsername")
                return render_template('userprofile.html', error="Username Changed")
            else:
                return render_template('userprofile.html', error_username = "Choose a different username, there is already another user with the same username")
        elif(request.form.get("newPassword") == request.form.get("confirmPassword") and request.form.get("newPassword") != None):
            cur.execute("UPDATE users SET hash=(?) WHERE email=(?)", (generate_password_hash(request.form.get("newPassword")), session["user_email"]))
            conn.commit()
            return render_template('userprofile.html', error="Password Changed")
        elif(request.form.get("newPassword") != request.form.get("confirmPassword") and request.form.get("newPassword") != None):
            return render_template('userprofile.html', error_password="Password Did Not Match")
        elif(request.form.get("deleteProfileButton")):
            cur.execute("DELETE FROM users WHERE email=(?)", session["user_email"])
            conn.commit()
            session["user_email"] = None
            session["user_password"] = None
            return render_template("index.html", error="")
        
@app.route("/resultlog", methods = ['GET', 'POST'])
def resultlog():
    if request.method == "GET":
        temp = json.loads(cur.execute("SELECT * FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
        return render_template('resultlog.html', data=temp)
    if request.method == "POST":
        if request.form.get("checklist"):
            return render_template("checklist.html", symptom_list = x.columns)
        elif request.form.get("result_log"):
            print(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            temp = json.loads(cur.execute("SELECT prevConditions FROM USERS WHERE email=(?)", session["user_email"]).fetchall()[0][0])
            return render_template('resultlog.html', data=temp)
        elif request.form.get("user_profile"):
            return render_template("userprofile.html")
        elif request.form.get("logout"):
            session.clear()
            redirect("/")
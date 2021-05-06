from flask import Flask, render_template, request, redirect, url_for, session
from sqlalchemy import create_engine , MetaData,Table,Column,Integer,String
import sqlalchemy
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
app.secret_key = 'a'

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/signup1')
def home1():
    return render_template("sign_up.html")

@app.route('/login1',methods=['POST','GET'])
def login1():
    return render_template("login.html")

@app.route('/signup',methods=["POST","GET"])
def signup():
    msg=""
    if request.method == 'POST':
            name1=request.form['username']
            email1=request.form['email']
            pass1=request.form['password1']
            pass2=request.form['password2']
            print(name1,email1,pass1,pass2)
            session["name"]=name1
            session["email"]=email1
            session["password1"]=pass1
            session["password2"]=pass2
    print(session["name"],session["email"],session["password1"])
    engine = create_engine('sqlite:///database2.db',echo=True)
    meta = MetaData()
    users = Table(
        'users',meta,
        Column('id',Integer,primary_key=True),
        Column('username',String),
        Column('email',String),
        Column('password',String)
    )
    meta.create_all(engine)
    conn  =engine.connect()
    res1 = conn.execute("SELECT * FROM users WHERE username=? ",(str(session["name"])))
    res2 = conn.execute("SELECT * FROM users WHERE email=?",(str(session["email"])))
    l1=len(list(res1.fetchall()))
    l2=len(list(res2.fetchall()))
    if(l1>0):
        msg = "Username  Already Exists"
    elif(l2>0):
        msg = "Email Already Exists"
    else:
        if(str(session['password1'])!=str(session['password2'])):
            msg = "Password doesn't matched"
        else:
            insert1 = users.insert()
            insert1 = users.insert().values(username = session["name"],email = session["email"],password=session["password1"])
            result = conn.execute(insert1)
            print("result :",result.inserted_primary_key[0])
            session.pop("name")
            session.pop("email")
            session.pop("password1")
            session.pop("password2")
            msg = 'You have successfully registered !'
    return render_template("sign_up.html",msg = msg)

@app.route('/login',methods=["POST","GET"])
def login():
    msg1=''
    if request.method == 'POST':
        print(request.form['email'])
        email1=request.form['email']
        pass1=request.form['password']
        engine = create_engine('sqlite:///database2.db',echo=True)
        conn = engine.connect()
        res1 = conn.execute("SELECT * FROM users WHERE email=?",(str(email1)))
        res2 = conn.execute("SELECT * FROM users WHERE email=? AND password = ?",(str(email1),str(pass1)))
        list1=list(res1.fetchall())
        len1 = len(list1)
        if(len1==1 and len(list(res2.fetchall()))==0):
            msg1="Password is incorrect"
            return render_template("login.html",msg1=msg1)
        elif(len1>0):
            return redirect(url_for('main'))
        else:
            msg1="User doesn't exists"
            return render_template("login.html",msg1=msg1)
        

@app.route('/main',methods=['POST','GET'])
def main():
    return render_template("main.html")

@app.route('/main_predict',methods=['POST','GET'])
def main1():
    msg=''
    if request.method == 'POST':
        gen_biomass=request.form['gen_biomass']
        gen_fosil_coal=request.form['gen_fosil_coal']
        gen_fosil_gas=request.form['gen_fosil_gas']
        for_wind_day_ahead=request.form['for_wind_day_ahead']

        gen_fosil_hard_coal=request.form['gen_fosil_hard_coal']
        gen_fosill_oil=request.form['gen_fosill_oil']
        gen_hydro_storage=request.form['gen_hydro_storage']
        price_ahead=request.form['price_ahead']

        gen_hydro_run_river=request.form['gen_hydro_run_river']
        gen_hydro_water=request.form['gen_hydro_water']
        gen_nuclear=request.form['gen_nuclear']
        price=request.form['price']

        gen_other=request.form['gen_other']
        gen_other_renewable=request.form['gen_other_renewable']
        gen_solar=request.form['gen_solar']
        year=request.form['year']

        gen_waste=request.form['gen_waste']
        gen_wind_onshore=request.form['gen_wind_onshore']
        for_solar_day_ahead=request.form['for_solar_day_ahead']
        month=request.form['month']
        week=request.form['week']

        print(gen_biomass,gen_fosil_coal,gen_fosil_gas,for_wind_day_ahead)
        print(gen_fosil_hard_coal,gen_fosill_oil,gen_hydro_storage,price_ahead)
        print(gen_hydro_run_river,gen_hydro_water,gen_nuclear,price)
        print(gen_other,gen_other_renewable,gen_solar,year)
        print(gen_waste,gen_wind_onshore,for_solar_day_ahead,month,week)


        test_list=[gen_biomass,gen_fosil_coal,gen_fosil_gas,gen_fosil_hard_coal,gen_fosill_oil,gen_hydro_storage
        ,gen_hydro_run_river,gen_hydro_water,gen_nuclear,gen_other,gen_other_renewable,gen_solar
        ,gen_waste,gen_wind_onshore,for_solar_day_ahead,for_wind_day_ahead,price_ahead,price,year,month,week]
        test_list2=list(list(map(float,test_list)))
        test_list1=np.array([test_list2])

        try:
            with open('model_pickle','rb') as f:
                model = pickle.load(f)
            y_pred_model = model.predict(test_list1)
            print("Result will be shown after input everything is fine : ",y_pred_model)
        except:
            cnx = create_engine('sqlite:///database1.db').connect()
            conn = cnx.connect()
            df = pd.read_sql_table('mains', cnx)
            x_train,x_test,y_train,y_test = train_test_split(df.drop(["total load actual","total load forecast","price day ahead"],axis=1),df["total load actual"],test_size=0.2,random_state=1)
            l1 = LinearRegression()
            l1.fit(x_train,y_train)
            y_pred1 = l1.predict(test_list1)
            y_pred2 = l1.predict(x_test)
            met1 = metrics.r2_score(y_test,y_pred2)
            print("metrics : ",met1)
            conn.execute('INSERT INTO metrics (accuracy) VALUES(?)',str(round(met1,4)*100))
            with open('model_pickle','wb') as f:
                pickle.dump(l1,f)
            pred="This is the predicted Load Value : "+str(y_pred1)+" with an Accuracy of "+str(round(met1,4)*100)
            return render_template("main.html",pred=pred)
            

        test_list_test=[gen_biomass,gen_fosil_coal,gen_fosil_gas,gen_fosil_hard_coal,gen_fosill_oil,gen_hydro_storage
        ,gen_hydro_run_river,gen_hydro_water,gen_nuclear,gen_other,gen_other_renewable,gen_solar
        ,gen_waste,gen_wind_onshore,for_solar_day_ahead,for_wind_day_ahead,y_pred_model,y_pred_model,price_ahead,price,year,month,week]

        cnx = create_engine('sqlite:///database1.db').connect()
        conn = cnx.connect()

        df1 = pd.read_sql_table('metrics', cnx)
        acc = conn.execute('SELECT * FROM metrics')
        acc_real = float(*(list(acc.fetchall())[-1]))

        df = pd.read_sql_table('mains', cnx).drop('id',axis=1)
        a_series = pd.Series(test_list_test,index=df.columns)
        df=df.append(a_series,ignore_index=True)

        x_train,x_test,y_train,y_test = train_test_split(df.drop(["total load actual","total load forecast","price day ahead"],axis=1),df["total load actual"],test_size=0.2,random_state=1)
        l = LinearRegression()
        l.fit(x_train,y_train)
        y_pred = l.predict(x_test)
        met = metrics.r2_score(y_test,y_pred)

        if(acc_real < round(met,4)*100):
            conn.execute('INSERT INTO metrics (accuracy) VALUES(?)',str(round(met,4)*100))
            print("inside")
        
            conn.execute("INSERT INTO mains (`generation biomass`, `generation fossil brown coal/lignite`, `generation fossil gas`, `generation fossil hard coal`, `generation fossil oil`, `generation hydro pumped storage consumption`, `generation hydro run-of-river and poundage`, `generation hydro water reservoir`, `generation nuclear`, `generation other`, `generation other renewable`, `generation solar`, `generation waste`, `generation wind onshore`, `forecast solar day ahead`, `forecast wind onshore day ahead`, `total load forecast`, `total load actual`, `price day ahead`, `price actual`, year, month, weekdays) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (float(gen_biomass),float(gen_fosil_coal),float(gen_fosil_gas),float(gen_fosil_hard_coal),float(gen_fosill_oil),
            float(gen_hydro_storage),float(gen_hydro_run_river),float(gen_hydro_water),float(gen_nuclear),float(gen_other),
            float(gen_other_renewable),float(gen_solar),float(gen_waste),float(gen_wind_onshore),float(for_solar_day_ahead),
            float(for_wind_day_ahead),float(y_pred_model),float(y_pred_model),float(price_ahead),float(price),year,month,week))
            
            df = pd.read_sql_table('mains', cnx)
            print(df)

            with open('model_pickle','wb') as f:
                pickle.dump(l,f)
   
    pred="This is the predicted Load Value : "+str(round(*y_pred_model,2))+" with an Accuracy of "+str(acc_real)
    return render_template("main.html",pred=pred)


if(__name__ == "__main__"):
    app.run()
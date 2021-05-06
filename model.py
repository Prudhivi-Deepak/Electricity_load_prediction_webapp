# import pandas as pd
import pickle
# from sqlalchemy import create_engine
# engine  = create_engine('sqlite:///database1.db')

# def change(x):
#     list1 = ['Sunday', 'Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday']
#     if x in list1:
#         return list1.index(x)
#     return x

# data_df = pd.read_csv(r"energy_dataset.csv")
# df1 = data_df.drop(['generation hydro pumped storage aggregated','forecast wind offshore eday ahead'],axis=1)
# df2 = df1.dropna()
# df2["time"] = pd.to_datetime(df2["time"],utc = True)
# df2["year"] = df2["time"].dt.year
# df2["month"] = df2["time"].dt.month
# df2["month_name"] = df2["time"].dt.month_name()
# df2["weekdays"] = df2["time"].dt.day_name()
# # df2["week"] = df2["time"].dt.week
# df2["weekdays"] = list(map(change,df2["weekdays"]))
# df2=df2.drop(["time","month_name",'generation fossil coal-derived gas','generation fossil oil shale','generation fossil peat','generation geothermal','generation marine','generation wind offshore'],axis=1)
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.linear_model import LinearRegression

# x_train,x_test,y_train,y_test = train_test_split(df2.drop(["total load actual","total load forecast","price day ahead"],axis=1),df2["total load actual"],test_size=0.2,random_state=1)
# l = LinearRegression()
# l.fit(x_train,y_train)
# y_pred = l.predict(x_test)
# print(metrics.r2_score(y_test,y_pred))

# with open('model_pickle','wb') as f:
#             pickle.dump(l,f)

with open('model_pickle','rb') as f:
    lr=pickle.load(f)
test =[[447.0,	329.0,	4844.0,	4821.0	,162.0	,863.0,	1051.0,	1899.0,	7096.0,	43.0,73.0,49.0,	196.0,6378.0,17.0,6436.0,65.41,2014,12,3]]
test1=[[301.0,	0.0,	6002.0,	1994.0,	217.0	,187.0,	1026.0,	1480.0,	6072.0,	56.0,	93.0,	14.0,	299.0	,3231.0,	20.0,	3408.0,	60.7,	2018,	12,	1,]]
pre = lr.predict(test1)
print(pre)




# filename = 'finalized_model.sav'
# pickle.dump(l, open(filename, 'wb'))
 
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(x_test, y_test)
# print(result)

# df2.to_sql('tbl_name', con=engine,index=False,if_exists='replace')
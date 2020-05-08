import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

df = pd.read_csv('/home/yuanjiexia/database/openmrs.csv', low_memory=False)
y_test_pred=[]

for i in tqdm(range(9600)):
    afterDrop=df.drop([i])
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
    m.fit(afterDrop.drop(columns=['y1']), afterDrop['y1'])
    x_test=df._slice(slice(i,i+1),0)
    x_test=x_test.drop(columns=['y1'])
    y_test_pred.append(m.predict(x_test))

f = open("/home/yuanjiexia/openmrs_demo.txt", "w")
f.writelines(["%s\n" % item  for item in y_test_pred])
f.close()
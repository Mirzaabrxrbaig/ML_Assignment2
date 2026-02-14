
import pandas as pd
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = pd.read_csv("Customer-Churn-Records.csv")

data = data.drop(columns=["RowNumber","CustomerId","Surname"], errors="ignore")
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Exited", axis=1)
y = data["Exited"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

models = {
    "logistic": Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000))]),
    "decision_tree": DecisionTreeClassifier(max_depth=6),
    "knn": Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=7))]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

os.makedirs("saved_models",exist_ok=True)

results=[]

for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]

    results.append([
        name,
        accuracy_score(y_test,pred),
        roc_auc_score(y_test,prob),
        precision_score(y_test,pred),
        recall_score(y_test,pred),
        f1_score(y_test,pred),
        matthews_corrcoef(y_test,pred)
    ])

    joblib.dump(model,f"saved_models/{name}.pkl")

cols=["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]
print(pd.DataFrame(results,columns=cols))

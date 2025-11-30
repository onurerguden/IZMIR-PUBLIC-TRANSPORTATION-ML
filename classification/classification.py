import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os
import seaborn as sns
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df=pd.read_csv("izmirim-kart-ulasim-istatistikleri-guncel.csv",sep=",")
df["DATE"]=pd.to_datetime(df["DATE"],format="%d.%m.%Y")
df = df.sort_values("DATE")
df["DAY_OF_WEEK"]=df["DATE"].dt.dayofweek
passenger_columns=["FULL_FARE", "STUDENT", "TEACHER", "SIXTY_YEARS_OLD",
                   "TICKET", "CHILD", "PERSONNEL", "FREE", "BANK CARD"]
df["TOTAL_PASSENGERS"]=df[passenger_columns].sum(axis=1)
df["PASSENGER_LEVEL"]=pd.qcut(df["TOTAL_PASSENGERS"],q=3,labels=["LOW","MEDIUM","HIGH"])

df=pd.get_dummies(df,columns=["INSTITUTION"]) #institution one hot encoded

y=df["PASSENGER_LEVEL"]
dropped_columns=["TOTAL_PASSENGERS","PASSENGER_LEVEL","DATE",]+passenger_columns
x=df.drop(dropped_columns,axis=1)

#splitting the dataset
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
train_size = int(len(df) * 0.8)

x_train = x.iloc[:train_size]
x_test = x.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
# 1-Decision tree algorithm

dt_model=DecisionTreeClassifier(criterion="gini",max_depth=10,random_state=42)

#training
dt_model.fit(x_train,y_train)

# decision tree prediction
dt_y_predicted=dt_model.predict(x_test)
dt_accuracy=accuracy_score(y_test,dt_y_predicted)
print("Decision tree Accuracy: ", dt_accuracy)
#print(classification_report(y_test,y_predicted))


#desicion tree plotting
output_dir="../plots/classification"

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=x.columns.tolist(),
    class_names=["LOW","MEDIUM","HIGH"],
    filled=True,
    rounded=True,
    fontsize=10,
    precision=2,
    proportion=True,
    impurity=True
)
plt.tight_layout()
save_path = os.path.join(output_dir, "decision_tree.png")
plt.savefig(save_path, dpi=300)
plt.close()
#plt.show()

# 2-SVM algorithm

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
svm_model=SVC(kernel="rbf",C=1,gamma="scale")
svm_model.fit(x_train_scaled,y_train)

#SVM prediction
svm_y_predicted=svm_model.predict(x_test_scaled)
svm_accuracy=accuracy_score(y_test, svm_y_predicted)
print("SVM Accuracy:",svm_accuracy)

# 3-Naive bayes algorithm

nb_model=GaussianNB()
nb_model.fit(x_train,y_train)
#bayes prediction
nb_y_predicted=nb_model.predict(x_test)
nb_accuracy=accuracy_score(y_test, nb_y_predicted)
print("Naive Bayes Accuracy:", nb_accuracy)

# 4-KNN algorithm

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled,y_train)
knn_y_predicted=knn.predict(x_test_scaled)
knn_accuracy=accuracy_score(y_test, knn_y_predicted)
print("KNN Accuracy:", knn_accuracy)

#plotting bar chart for algorithm accuracy comparison


accuracies = {
    "Decision Tree": dt_accuracy,
    "SVM": svm_accuracy,
    "Naive Bayes": nb_accuracy,
    "KNN": knn_accuracy
}

models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=scores,hue=models, palette="Blues_r",width=0.3)

plt.title("Model Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy scale 0–1
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
plt.close()

print("Accuracy comparison plot saved to:", os.path.join(output_dir, "accuracy_comparison.png"))



#------------------------------------------------------
param_grid = {
    'max_depth': [5, 7, 10, 15],  # Ağacın maksimum derinliği
    'min_samples_leaf': [1, 5, 10], # Bir yaprak düğümdeki minimum örnek sayısı
    'criterion': ['gini', 'entropy'] # Bölme kriteri
}

# 2. Karar Ağacı Modelini Tanımlama
dt = DecisionTreeClassifier(random_state=42)

# 3. TimeSeriesSplit ile Çapraz Doğrulama Stratejisini Tanımlama
# Zaman serisi verisi olduğu için bu şart.
# 5 farklı eğitim/test seti oluşturacak.
tscv = TimeSeriesSplit(n_splits=5)

# 4. GridSearchCV'yi Kurma
# cv=tscv: Çapraz doğrulama için TimeSeriesSplit kullan
# scoring='accuracy': Performans metriği olarak doğruluğu kullan
# n_jobs=-1: Tüm CPU çekirdeklerini kullanarak hesaplamayı hızlandır
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)

# 5. Grid Search'ü Eğitim Verisi Üzerinde Çalıştırma
print("Decision Tree için Grid Search başlıyor...")
grid_search.fit(x_train, y_train)
print("Grid Search tamamlandı.")

# --- SONUÇLARI ALMA ---

# En iyi modeli ve parametreleri çekme
best_dt_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Test seti üzerinde en iyi modelin performansını ölçme
dt_y_predicted_optimized = best_dt_model.predict(x_test)
dt_accuracy_optimized = accuracy_score(y_test, dt_y_predicted_optimized)


print("\n--- Decision Tree Grid Search Sonuçları ---")
print(f"En İyi Parametreler: {best_params}")
print(f"Optimize Edilmiş DT Doğruluğu (Test Seti): {dt_accuracy_optimized:.4f}")





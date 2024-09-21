#Ricordati sempre di eseguire questi comandi sul terminale
# pip install numpy pandas matplotlib seaborn scikit-learn scipy
#Maggiori info nel README.md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import scipy.stats as stats
from sklearn import metrics

# STEP 1: Selezionare e caricare il dataset
dati = pd.read_csv("data/Fuel_Consumption Ratings_2023.csv", encoding='latin-1')

# Stampiamo le colonne del dataset
print()
print("Info dataset: ")
print(dati.info())

# Stampiamo le prime righe del dataset
print(dati.head())

# STEP 2: Pre-Processing
# Rimuoviamo le colonne inutili
dati = dati.drop(columns=["Transmission", "Comb(mpg)", "Year"])
print(dati.info())

# Rimuoviamo eventuali NaN
dati = dati.dropna()
print()
print("Info dataset dopo rimozione NaN: ")
print(dati.info())

# Controlliamo i descrittori statistici delle variabili numeriche per identificare eventuali valori fuori soglia
print(dati.describe())

# Utilizzo boxplot per visualizzare meglio la distribuzione delle variabili numeriche
for colonna in dati.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(dati[colonna])
    plt.title(f'Distribuzione di {colonna}')
    plt.show()

# STEP 3: Exploratory Data Analysis (EDA)
# Selezioniamo solo le colonne numeriche per la matrice di correlazione
dati_numerici = dati.select_dtypes(include=[np.number])

# Matrice di correlazione
matrice_correlazione = dati_numerici.corr()

# Visualizziamo la matrice di correlazione
plt.figure(figsize=(12, 8))
sns.heatmap(matrice_correlazione, annot=True, cmap='coolwarm')
plt.title('Matrice di Correlazione')
plt.show()

# Visualizziamo la matrice di correlazione usando l'alternativa vista a lezione
#plt.figure(figsize=(12, 8))
#plt.matshow(correlation_matrix, vmin=-1, vmax=1, fignum=1)
#plt.xticks(np.arange(0, numeric_data.shape[1]), numeric_data.columns, rotation=45)
#plt.yticks(np.arange(0, numeric_data.shape[1]), numeric_data.columns)
#plt.title("Matrice di correlazione")
#plt.colorbar()
#plt.show()

# Analisi univariate - Istogrammi delle variabili numeriche
dati_numerici.hist(bins=30, figsize=(20, 15), color='blue', edgecolor='black')
plt.suptitle('Istogrammi delle variabili numeriche')
plt.show()

# Analisi bivariate - Scatter plot tra variabili fortemente correlate
# Selezioniamo la seconda coppia di variabili con la correlazione più alta
coppie_alta_correlazione = [('Comb (L/100 km)', 'CO2 Emissions (g/km)')]
print("Coppia di variabili con alta correlazione:", coppie_alta_correlazione)


# Visualizziamo gli scatter plot per la coppia selezionata
for (var1, var2) in coppie_alta_correlazione:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=dati_numerici[var1], y=dati_numerici[var2])
    plt.title(f'Scatter plot tra {var1} e {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.show()
    
    
# Analisi multivariata - Pair plot per visualizzare tutte le relazioni bivariate tra le variabili più rilevanti
# Selezioniamo le variabili con la correlazione più alta rispetto a 'CO2 Emissions (g/km)'
correlazioni = matrice_correlazione['CO2 Emissions (g/km)'].abs().sort_values(ascending=False)
soglia_correlazione = 0.8

# Variabili rilevanti (escludiamo Fuel Consumption e includiamo Comb)
variabili_rilevanti = [var for var in correlazioni.index if correlazioni[var] > soglia_correlazione and var not in ['Hwy (L/100 km)', 'Fuel Consumption (L/100Km)']]
if 'CO2 Emissions (g/km)' not in variabili_rilevanti:
    variabili_rilevanti.append('CO2 Emissions (g/km)')

# Creiamo il pair plot solo con le variabili rilevanti
sns.pairplot(dati_numerici[variabili_rilevanti])
plt.suptitle('Pair Plot delle Variabili Numeriche Rilevanti', y=1.02)
plt.show()


# STEP 4: Splitting
# Definire le feature (X) e il target (y)
target = 'CO2 Emissions (g/km)'
X = dati_numerici.drop(columns=[target])
y = dati_numerici[target]

# Dividiamo il dataset in 70% training, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# STEP 5: Regressione
# Utilizziamo la coppia di variabili con la seconda correlazione più alta
var1, var2 = coppie_alta_correlazione[0]
X_reg = dati_numerici[[var1]].values.reshape(-1, 1)
y_reg = dati_numerici[var2].values

# Creiamo il modello di regressione lineare
modello_regressione = LinearRegression()
modello_regressione.fit(X_reg, y_reg)

# Predizioni
y_pred = modello_regressione.predict(X_reg)

# Coefficienti della regressione
coefficiente = modello_regressione.coef_[0]
intercetta = modello_regressione.intercept_

# Calcolo del coefficiente di determinazione R^2
r2 = modello_regressione.score(X_reg, y_reg)

# Calcolo del Mean Squared Error (MSE)
mse = mean_squared_error(y_reg, y_pred)

# Analisi di normalità dei residui
residui = y_reg - y_pred
(mu, sigma) = stats.norm.fit(residui)
plt.figure(figsize=(10, 6))
sns.histplot(residui, kde=True, stat="density")
plt.title(f'Analisi di normalità dei residui per {var1} vs {var2}')
plt.xlabel('Residui')
plt.ylabel('Densità')
plt.show()

# Visualizziamo i risultati della regressione
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='blue', label='Dati')
plt.plot(X_reg, y_pred, color='red', linewidth=2, label='Retta di regressione')
plt.title(f'Regressione Lineare tra {var1} e {var2}')
plt.xlabel(var1)
plt.ylabel(var2)
plt.legend()
plt.show()

print()
print(f"Regressione tra {var1} e {var2}:")
print(f"Coefficiente: {coefficiente}")
print(f"Intercetta: {intercetta}")
print(f"R^2: {r2}")
print(f"MSE: {mse}")
print("\n")

# Ripetiamo per correlazione negativa alta 
coppie_negativa_correlazione = [('CO2 Emissions (g/km)', 'CO2 Rating')]
var1, var2 = coppie_negativa_correlazione[0]
X_reg = dati_numerici[[var1]].values.reshape(-1, 1)
y_reg = dati_numerici[var2].values

# Creiamo il modello di regressione lineare
modello_regressione = LinearRegression()
modello_regressione.fit(X_reg, y_reg)

# Predizioni
y_pred = modello_regressione.predict(X_reg)

# Coefficienti della regressione
coefficiente = modello_regressione.coef_[0]
intercetta = modello_regressione.intercept_

# Calcolo del coefficiente di determinazione R^2
r2 = modello_regressione.score(X_reg, y_reg)

# Calcolo del Mean Squared Error (MSE)
mse = mean_squared_error(y_reg, y_pred)

# Analisi di normalità dei residui
residui = y_reg - y_pred
(mu, sigma) = stats.norm.fit(residui)
plt.figure(figsize=(10, 6))
sns.histplot(residui, kde=True, stat="density")
plt.title(f'Analisi di normalità dei residui per {var1} vs {var2}')
plt.xlabel('Residui')
plt.ylabel('Densità')
plt.show()

# Visualizziamo i risultati della regressione
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='blue', label='Dati')
plt.plot(X_reg, y_pred, color='red', linewidth=2, label='Retta di regressione')
plt.title(f'Regressione Lineare tra {var1} e {var2}')
plt.xlabel(var1)
plt.ylabel(var2)
plt.legend()
plt.show()

print()
print(f"Regressione tra {var1} e {var2}:")
print(f"Coefficiente: {coefficiente}")
print(f"Intercetta: {intercetta}")
print(f"R^2: {r2}")
print(f"MSE: {mse}")
print("\n")



# STEP 6: Addestramento del Modello - Classificazione con Regressione Logistica e SVM

# Trasformazione del target in binario
# Supponiamo di voler classificare CO2 Emissions (g/km) sopra o sotto la media
y_binary = (y > y.mean()).astype(int)

# Utilizza i set di training, validation e test già definiti
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Selezioniamo due feature per la visualizzazione
col1, col2 = 'Comb (L/100 km)', 'Engine Size (L)'
X_train_2d = X_train[[col1, col2]]
X_val_2d = X_val[[col1, col2]]

# Addestramento del modello di Regressione Logistica sul training set
logistic_model = LogisticRegression()
logistic_model.fit(X_train_2d, y_train)

# Predizioni sul validation set
y_val_pred_log = logistic_model.predict(X_val_2d)

# Valutazione del modello di Regressione Logistica sul validation set
print()
print("Valutazione del modello di Regressione Logistica (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_log)}")
print(classification_report(y_val, y_val_pred_log, zero_division=0))

# Matrice di Confusione per Regressione Logistica sul validation set
conf_matrix_log_val = metrics.confusion_matrix(y_val, y_val_pred_log)
cm_display_log_val = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_log_val, display_labels=[False, True])
cm_display_log_val.plot()
plt.title('Matrice di Confusione - Regressione Logistica (Validation Set)')
plt.show()

# Addestramento del modello SVM con kernel lineare sul training set
modello_svm_linear = SVC(kernel='linear', C=1)
modello_svm_linear.fit(X_train_2d, y_train)

# Predizioni sul validation set
y_val_pred_svm_linear = modello_svm_linear.predict(X_val_2d)

# Valutazione del modello SVM con kernel lineare sul validation set
print()
print("Valutazione del modello SVM con kernel lineare (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_svm_linear)}")
print(classification_report(y_val, y_val_pred_svm_linear))

# Matrice di Confusione per SVM con kernel lineare sul validation set
conf_matrix_svm_linear_val = metrics.confusion_matrix(y_val, y_val_pred_svm_linear)
cm_display_svm_linear_val = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_linear_val, display_labels=[False, True])
cm_display_svm_linear_val.plot()
plt.title('Matrice di Confusione - SVM con kernel lineare (Validation Set)')
plt.show()

# Addestramento del modello SVM con kernel RBF sul training set
modello_svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
modello_svm_rbf.fit(X_train_2d, y_train)

# Predizioni sul validation set
y_val_pred_svm_rbf = modello_svm_rbf.predict(X_val_2d)

# Valutazione del modello SVM con kernel RBF sul validation set
print()
print("Valutazione del modello SVM con kernel RBF (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_svm_rbf)}")
print(classification_report(y_val, y_val_pred_svm_rbf))

# Matrice di Confusione per SVM con kernel RBF sul validation set
conf_matrix_svm_rbf_val = metrics.confusion_matrix(y_val, y_val_pred_svm_rbf)
cm_display_svm_rbf_val = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_rbf_val, display_labels=[False, True])
cm_display_svm_rbf_val.plot()
plt.title('Matrice di Confusione - SVM con kernel RBF (Validation Set)')
plt.show()

# Visualizzazione delle previsioni di classificazione per SVM con kernel lineare
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_val_2d[col1], y=X_val_2d[col2], hue=y_val_pred_svm_linear, palette='coolwarm', marker='s', alpha=0.7)
plt.title('Grafico di separazione e Classificazione con SVM (kernel lineare)')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(title='Classe', loc='upper left')

# Creazione della griglia per la visualizzazione della decision boundary
xx, yy = np.meshgrid(np.linspace(X_val_2d[col1].min(), X_val_2d[col1].max(), 100),
                     np.linspace(X_val_2d[col2].min(), X_val_2d[col2].max(), 100))
Z = modello_svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.show()

# Visualizzazione delle previsioni di classificazione per SVM con kernel RBF
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_val_2d[col1], y=X_val_2d[col2], hue=y_val_pred_svm_rbf, palette='coolwarm', marker='s', alpha=0.7)
plt.title('Grafico di separazione e Classificazione con SVM (kernel RBF)')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(title='Classe', loc='upper left')

# Creazione della griglia per la visualizzazione della decision boundary
xx, yy = np.meshgrid(np.linspace(X_val_2d[col1].min(), X_val_2d[col1].max(), 100),
                     np.linspace(X_val_2d[col2].min(), X_val_2d[col2].max(), 100))
Z = modello_svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.show()



# STEP 7: Hyperparameter Tuning
# Definiamo la griglia di parametri per SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Configura la Grid Search con la validazione incrociata
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)

# Esegui la Grid Search sul training set
grid_search.fit(X_train, y_train)

# Mostra i migliori iperparametri trovati
print(f"I migliori iperparametri sono: {grid_search.best_params_}")

# Valuta il modello ottimale sul validation set
best_svm_model = grid_search.best_estimator_
y_val_pred_best_svm = best_svm_model.predict(X_val)

print()
print("Valutazione del modello ottimizzato SVM (Validation Set):")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_best_svm)}")
print(classification_report(y_val, y_val_pred_best_svm))

# Matrice di Confusione per SVM ottimizzato sul validation set
conf_matrix_best_svm_val = metrics.confusion_matrix(y_val, y_val_pred_best_svm)
cm_display_best_svm_val = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best_svm_val, display_labels=[False, True])
cm_display_best_svm_val.plot()
plt.title('Matrice di Confusione - SVM Ottimizzato (Validation Set)')
plt.show()

#Sulle slide del corso è presente anche il k-fold Cross-Validation, ma siccome nella
#consegna non è richiesto e siccome non lo abbiamo visto a lezione, non lo trattiamo

#Non è richiesta neanche la realizzazione del grafico di separazione, poichè 
#il nostro obiettivo  è predirre un dato numeri, non vogliami separare i dati

# STEP 8: Valutazione della Performance
# Valutazione finale sul test set
y_test_pred_best_svm = best_svm_model.predict(X_test)
print()
print("Valutazione finale del modello ottimizzato SVM sul test set:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_best_svm)}")
print(classification_report(y_test, y_test_pred_best_svm))

# Matrice di Confusione finale per SVM ottimizzato sul test set
conf_matrix_test_best_svm = metrics.confusion_matrix(y_test, y_test_pred_best_svm)
cm_display_test_best_svm = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test_best_svm, display_labels=[False, True])
cm_display_test_best_svm.plot()
plt.title('Matrice di Confusione - SVM Ottimizzato (Test Set)')
plt.show()



# STEP 9: Studio statistico sui risultati della valutazione
# Ripetiamo l'addestramento e il testing k volte (con k ≥ 10) per valutare la robustezza del modello
k = 10
accuratezze = []

# Usare i migliori parametri trovati nel punto 7 per il modello
best_params_cv = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

for i in range(k):
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=i)
    modello_svm = SVC(**best_params_cv)
    modello_svm.fit(X_train, y_train)
    predizioni = modello_svm.predict(X_test)
    acc = accuracy_score(y_test, predizioni)
    accuratezze.append(acc)
    
# Analisi statistica descrittiva delle metriche di errore
accuratezze = np.array(accuratezze)
media_acc = np.mean(accuratezze)
std_acc = np.std(accuratezze)
intervallo_confidenza = stats.norm.interval(0.95, loc=media_acc, scale=std_acc/np.sqrt(k))

print()
print(f"Media Accuracy: {media_acc}")
print(f"Deviazione Standard: {std_acc}")
print(f"Intervallo di Confidenza al 95%: {intervallo_confidenza}")

# Istogramma e boxplot delle accuratezze
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(accuratezze, bins=10, edgecolor='black')
plt.title('Distribuzione delle Accuratezze')
plt.xlabel('Accuracy')
plt.ylabel('Frequenza')

plt.subplot(1, 2, 2)
plt.boxplot(accuratezze, vert=False)
plt.title('Boxplot delle Accuratezze')
plt.xlabel('Accuracy')
plt.xlim(0.98, 1.01) 

plt.tight_layout()
plt.show()
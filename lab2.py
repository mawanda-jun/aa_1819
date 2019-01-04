import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_svmlight_file
import urllib.request
# set splitting
from sklearn.model_selection import train_test_split
# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score
# learning
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# model selection
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics


def evaluate(y_target, y_pred):
    print('accuracy: {a}\nprecision: {p}\nrecall: {r}'.format(
        a=accuracy_score(y_target, y_pred),
        p=precision_score(y_target, y_pred),
        r=recall_score(y_target, y_pred)))


# breast = load_breast_cancer()

# X, y = breast.data, breast.target
# print(X.shape)  # (569, 30), 30 features, 569 examples

raw_data = urllib.request.urlopen('http://www.math.unipd.it/~mpolato/didattica/ml1819/tic-tac-toe.svmlight')

X, y = load_svmlight_file(raw_data, ) # rawdata contiene il file. Se invece il file e' dentro al computer, gli si mette il path. Torna X sparso!
X = X.toarray()
print(X.shape, y.shape)

# che percentuale teniamo per test/training?
# sklearn, module selection, offre il train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# preprocessing. MinMax scaler stanno tra un minimo e un massimo per normalizzare.
scaler = MinMaxScaler()   # non va bene applicare lo scaling a tutto il dataset! Dobbiamo applicarlo solamente
                            # il training
scaler.fit(X_train) # abbiamo allenato lo scaler per capire come scalarli
X_train = scaler.transform(X_train) # modifica X_train, trasformandolo con cio' che ha imparato col fit di prima
X_test = scaler.transform(X_test)   # modifica X_test ma allenato su X_train, cosi' non si bara.
###############################################################
# # training
# # trees
# clf_tree = tree.DecisionTreeClassifier()
# clf_tree = clf_tree.fit(X_train, y_train)
#
# y_pred = clf_tree.predict(X_test)
# print("Trees")
# evaluate(y_test, y_pred)
#
# # multilayer perceptron
# mlp = MLPClassifier(
#     hidden_layer_sizes=(100,),  # unico layer nascosto con 100 nodi
#     max_iter=500,
#     alpha=1,
# )
# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)
# print('MLP')
# evaluate(y_test, y_pred)
#
# # SVM
# svc = svm.SVC(
#     gamma=.0001,
#     C=1000.0,
# )
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
# print("SVM")
# evaluate(y_test, y_pred)
#############################################################
# come si fa model selection?
# come utilizzare cross-validation con test ripetuto
# creiamo una griglia per SVM: gamma, C, iperparametri kernel
p_grid = [
    {
        "C": [2**i for i in range(-3, 4)],
        'kernel':['rbf'],
        'gamma':[10**i for i in range(-5, 1)]
    },
    {
        "C": [2**i for i in range(-3, 4)],
        'kernel':['poly'],
        'degree':[i for i in range(2, 5)]
    }
]
# lista di dizionari. Ogni dizionario contiene uno dei parametri. Sarebe buona norma che se il miglior valore sta
# nel bordo della griglia allora conviene espanderla un po'.
# Possiamo anche definire un kernel definito da noi. In genere si fa a mano.

# valutazione ripetuta -> ci facciamo aiutare con KFold
# genera KFold!
# Farne di piu' rende il risultato piu' robusto. Divisione in fold serve per valutare il metodo che ho trovato con
# nuovi dati. Per ogni fold poi se lo divido
skf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# skf torna gli indici. Ci dobbiamo occupare di separare il dataset
accs = []
fold = 1

for train, test in skf.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    clf = GridSearchCV(
        svm.SVC(gamma='auto'),   # prende i parametri dalla griglia che gli passiamo. Se non glielo passiamo dobbiamo dirglielo noi.
        param_grid=p_grid,
        cv=3,  # qui in pratica si fa un nested kfold
        scoring='accuracy'
    )
    # what the hell is happening here??
    # il primo kfold divide train/test in 5 parti. Ne prende 4 e ne usa 1 per validation. Dopodiche' GridSearch ne fa un
    # nested-fold (cv=3) per poter valutare i vari modelli per ogni n-fold. Dopodiche' ripete la valutazione sugli altri fold
    # questo assicura una valutazione unbiased dal punto di vista statistico

    clf.fit(X_train, y_train)

    # cerchiamo il miglior modello! Vorremmo farci dare il valore della accuracy
    print('validation score: ', clf.best_score_)
    print('best hyper-parameters: ', clf.best_params_)

    # gridsearch una volta trovato il modello migliore fa l'allenamento su tutto il training set
    # dentro a clf ci sara' il miglior modello
    y_pred = clf.predict(X_test)

    # in classificazione e' utile vedere la matrice di confusione: magari ci sono alcune classi piu' difficili di altre
    print('report: ', metrics.classification_report(y_test, y_pred))
    print('Matrice di confusione: ', metrics.confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    print("accuracy fold {d}: {f}".format(d=fold, f=acc))
    fold += 1
print('AVG accuracy:', np.mean(accs), '+-', np.std(accs))




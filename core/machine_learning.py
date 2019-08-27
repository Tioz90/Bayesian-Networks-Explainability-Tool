import time
from pprint import pprint
import pandas as pd
import helper as hp

# used when formatting in f-strings
WIDTH = 0
PRECISION = 5



def linear():
	print("\n################## Linear Classifier ##################")

	from sklearn import linear_model

	clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

	return clf

def logistic():
	print("\n################## Logistic Regression ##################")

	from sklearn import linear_model

	clf = linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=2000)

	return clf

def lda():
	print("\n################## Linear Discriminant Analysis ##################")

	from sklearn import discriminant_analysis

	clf = discriminant_analysis.LinearDiscriminantAnalysis()

	return clf

def class_tree():
	print("\n################## Classification Tree ##################")

	from sklearn import tree

	clf = tree.DecisionTreeClassifier()

	return clf

def naif_bayes():
	print("\n################## Naive Bayes ##################")

	from sklearn import naive_bayes

	clf = naive_bayes.GaussianNB()

	return clf

def knn():
	print("\n################## K Nearest Neighbours ##################")

	from sklearn import neighbors

	clf = neighbors.KNeighborsClassifier()

	return clf

def svc():
	print("\n################## Support Vector Classifier ##################")

	from sklearn import svm

	clf = svm.SVC(gamma="scale", kernel="rbf")

	return clf

def rand_forest():
	print("\n################## Random Forest Classifier ##################")

	from sklearn import ensemble

	clf = ensemble.RandomForestClassifier(n_estimators=100)

	return clf

def adaboost():
	print("\n################## AdaBoost Classifier ##################")

	from sklearn import ensemble

	clf = ensemble.AdaBoostClassifier()

	return clf

def test_algorithms(X, y):

	from sklearn.model_selection import cross_val_score, cross_val_predict
	from sklearn.metrics import confusion_matrix
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import f_classif

	algorithms = {"Linear":              linear,
	              "Logistic":            logistic,
	              "LDA":                 lda,
	              "Classification Tree": class_tree,
	              "Naive Bayes":         naif_bayes,
	              "KNN":                 knn,
	              "SVC":                 svc,
	              "Random Forest":       rand_forest,
	              "AdaBoost":            adaboost}

	from collections import defaultdict

	scores_dict = defaultdict(dict)

	begin_time = time.time()

	for i in range(1, len(X.columns)):
	# for i in range(1, 3):

		# print("#################################################")
		print(f"\nNumber of features selected: {i}")
		# print("#################################################")
		# print(f"Features pre-selection: {list(regr_pn.columns)}")

		selected_columns = SelectKBest(score_func=f_classif, k=i).fit(X, y).get_support()
		X_red = X.loc[:, selected_columns]  # seleziona features

		print(f"Features post-selection: {hp.printl(list(X_red.columns))}")

		for alg in algorithms:
			# print(f"{alg}:")

			clf = algorithms[alg]()

			if alg not in scores_dict:
				scores_dict[alg] = defaultdict(list)

			scores_dict[alg]["Features"].append(X.columns[selected_columns])

			score = cross_val_score(clf, X_red, y, cv=10)
			scores_dict[alg]["Accuracy"].append(score.mean())
			scores_dict[alg]["Std Deviation"].append(score.std())
			print(f"Accuracy: {score.mean():{WIDTH}{PRECISION}} (+/- {score.std():{WIDTH}{PRECISION}})")

			y_pred = cross_val_predict(clf, X_red, y, cv=10) # TODO fa la predizione in cross validation
			scores_dict[alg]["Predictions"].append(y_pred)

			conf_mat = confusion_matrix(y, y_pred)
			scores_dict[alg]["Confusion Mx"].append(conf_mat)
			pprint(conf_mat)

	# TODO cost-based classification, puo' essere piu' importante prevedere un valore piuttosto che un altro
	# TODO prima di fare ML fare analisi dei dati per vedere come distribuite le features (?)
	# TODO AUC/ROC curves per capire meglio a prescindere da accuratezza
	# TODO usare rete bayesiana per spingere su explainability in ottica della mia tesi, apprendere la rete e testare, predirre la probabilita' delle features data la rete

	end_time = time.time()
	print(f"Time elapsed: {int(end_time - begin_time)} seconds")

	return scores_dict

def test_classification_algorithms(df, var_list):
	return_values = {}

	# scaling
	from sklearn import preprocessing

	for var in var_list:
		print("\nClassifying " + var)
		# print(f"Possible values: {hp.printl(df[target].unique())}")
		X = df.drop(columns=var)
		scaler = preprocessing.StandardScaler()
		X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)
		y = df[var]

		test_dict = test_algorithms(X, y)
		max_score, max_key, max_index = find_max_accuracy(test_dict)

		print_result(test_dict, max_score, max_key, max_index)
		# TODO ritornare i risultati in una lista e stampare a parte


def test_bayesian_network(var, df_codes, df_values, code_to_value_map, value_to_code_map):
	from sklearn.model_selection import StratifiedKFold
	import bayesian as bn
	import pandas as pd

	# None columns are marked for prediction
	X = df_codes.copy()
	y = X[var]

	skf = StratifiedKFold( n_splits=2, shuffle=True )
	skf.get_n_splits( X, y )

	train_index, test_index = skf.split( X, y )
	# print( "TRAIN:", len(train_index), "TEST:", len(test_index) )

	X.loc[:, var] = None
	train_split_0 = X.iloc[train_index[0]]
	test_split_0 = X.iloc[test_index[0]]
	train_split_1 = X.iloc[train_index[1]]
	test_split_1 = X.iloc[test_index[1]]


	# model = bn.BN( X, df_values, code_to_value_map, value_to_code_map )



	# run prediction
	# y_pred = pd.DataFrame( model.predict( X.values ), columns=X.columns )

	return


def print_result(dict, max_score, max_key, max_index):

	print(f"Best accuracy: {max_score:{WIDTH}.{PRECISION}} corresponding to {max_key} algorithm with {max_index + 1} features")
	print(f"Features selected are {hp.printl(dict[max_key]['Features'][max_index])}")
	print(f"Standard deviation: {dict[max_key]['Std Deviation'][max_index]:{WIDTH}.{PRECISION}}")
	print(f"Confusion matrix:\n {dict[max_key]['Confusion Mx'][max_index]}")

	return


def find_max_accuracy(dict):

	max_score = 0
	max_key = None
	max_index = 0
	for key in dict:
		if max(dict[key]["Accuracy"]) > max_score:
			max_score = max(dict[key]["Accuracy"])
			max_key = key
			max_index = dict[key]["Accuracy"].index(max_score)

	return max_score, max_key, max_index
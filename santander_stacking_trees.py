#python3

#Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn import pipeline, grid_search
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Gini and Entropy
from sklearn.ensemble import AdaBoostClassifier


from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score


from sklearn import svm

if __name__ == "__main__":
	
	start_time = time.time()
	print ("\n----- Reading data -----")
	print (round(((time.time() - start_time)/60),2))
	
	# Getting data ready. I'll use 4 Stratified CV to get probs of training data on 1st layer. 
	train_df = pd.read_csv("C:\\Users\\carrai1\\Desktop\\Projects\\Santander\\datasets\\train.csv")
	test_df = pd.read_csv("C:\\Users\\carrai1\\Desktop\\Projects\\Santander\\datasets\\test.csv")
	
	x_train_df = train_df.drop(["TARGET","ID"],axis=1)
	y_train = train_df["TARGET"]
	x_test_id = test_df["ID"]
	x_test = test_df.drop("ID", axis=1)
		
	################### Feature selection & feature engineering ###################################
	print ("\nScaling .... ")
	columns_train = x_train_df.columns 
	columns_test = x_test.columns  
	scale = StandardScaler()
	x_train_scale = pd.DataFrame(scale.fit_transform(x_train_df), columns=columns_train)
	x_test = pd.DataFrame(scale.transform(x_test), columns=columns_test)
	print(list(x_train_scale.columns.values))
	#PCA
	pca = PCA(n_components=4)
	x_train_projected = pca.fit_transform(x_train_scale)
	x_test_projected = pca.transform(x_test)
	x_train_scale.insert(1, 'PCAOne', x_train_projected[:, 0])
	x_train_scale.insert(1, 'PCATwo', x_train_projected[:, 1])
	x_train_scale.insert(1, 'PCAThree', x_train_projected[:, 2])
	x_train_scale.insert(1, 'PCAFour', x_train_projected[:, 3])
	x_test.insert(1, 'PCAOne', x_test_projected[:, 0])
	x_test.insert(1, 'PCATwo', x_test_projected[:, 1])
	x_test.insert(1, 'PCAThree', x_test_projected[:, 2])
	x_test.insert(1, 'PCAFour', x_test_projected[:, 3])		
	
	test_separation = x_train_df.shape[0]
	all_df = pd.concat((x_train_scale,x_test),axis=0,ignore_index=True)
		
	print ("\nRemoving features with no variance .... ")
	for feature in all_df.columns:
		if all_df[feature].std == 0 :
			print ("removing", feature)
			all_df.drop(feature, axis=1)
			x_test.drop(feature, axis=1)
	
	#remove constant columns
	print(all_df.shape)
	all_df = all_df.loc[:,all_df.apply(pd.Series.nunique) != 1]
	print(all_df.shape)	
	print(list(all_df.columns.values))
	
	all_df_copy = all_df.copy()
	all_df_copy.drop(['PCAOne', 'PCATwo', 'PCAThree','var36'], axis=1, inplace=True)
	
	all_df["std"] = all_df_copy.std(axis=1)
	all_df["sum_rows_scale"] = all_df_copy.sum(axis=1)

	all_df["negative"] = (all_df_copy < 0).astype(int).sum(axis=1)
	all_df["positive"] = (all_df_copy > 0).astype(int).sum(axis=1)

	all_df["poutliers_02"] = (all_df_copy > 0.2).astype(int).sum(axis=1)
	all_df["poutliers_03"] = (all_df_copy > 0.3).astype(int).sum(axis=1)
	all_df["poutliers_04"] = (all_df_copy > 0.4).astype(int).sum(axis=1)
	all_df["poutliers_05"] = (all_df_copy > 0.5).astype(int).sum(axis=1)
	all_df["poutliers_1"] = (all_df_copy > 1).astype(int).sum(axis=1)
	all_df["poutliers_2"] = (all_df_copy > 2).astype(int).sum(axis=1)
	all_df["poutliers_3"] = (all_df_copy > 3).astype(int).sum(axis=1)
	all_df["poutliers_5"] = (all_df_copy > 5).astype(int).sum(axis=1)
	all_df["poutliers_7"] = (all_df_copy > 7).astype(int).sum(axis=1)
	all_df["poutliers_10"] = (all_df_copy > 10).astype(int).sum(axis=1)
	all_df["poutliers_12"] = (all_df_copy > 12).astype(int).sum(axis=1)
	all_df["poutliers_15"] = (all_df_copy > 15).astype(int).sum(axis=1)
	all_df["poutliers_18"] = (all_df_copy > 18).astype(int).sum(axis=1)
	all_df["poutliers_20"] = (all_df_copy > 20).astype(int).sum(axis=1)
	all_df["poutliers_30"] = (all_df_copy > 30).astype(int).sum(axis=1)
	all_df["poutliers_50"] = (all_df_copy > 50).astype(int).sum(axis=1)
	all_df["poutliers_75"] = (all_df_copy > 75).astype(int).sum(axis=1)
	all_df["poutliers_100"] = (all_df_copy > 100).astype(int).sum(axis=1)
	all_df["poutliers_200"] = (all_df_copy > 200).astype(int).sum(axis=1)

	all_df["noutliers_02"] = (all_df_copy < -0.2).astype(int).sum(axis=1)
	all_df["noutliers_03"] = (all_df_copy < -0.3).astype(int).sum(axis=1)
	all_df["noutliers_04"] = (all_df_copy < -0.4).astype(int).sum(axis=1)
	all_df["noutliers_05"] = (all_df_copy < -0.5).astype(int).sum(axis=1)
	all_df["noutliers_1"] = (all_df_copy < -1).astype(int).sum(axis=1)
	all_df["noutliers_2"] = (all_df_copy < -2).astype(int).sum(axis=1)
	all_df["noutliers_3"] = (all_df_copy < -3).astype(int).sum(axis=1)
	all_df["noutliers_5"] = (all_df_copy < -5).astype(int).sum(axis=1)
	all_df["noutliers_7"] = (all_df_copy < -7).astype(int).sum(axis=1)
	all_df["nutliers_10"] = (all_df_copy < -10).astype(int).sum(axis=1)
	all_df["noutliers_12"] = (all_df_copy < -12).astype(int).sum(axis=1)
	all_df["noutliers_15"] = (all_df_copy < -15).astype(int).sum(axis=1)
	all_df["noutliers_18"] = (all_df_copy < -18).astype(int).sum(axis=1)
	all_df["noutliers_20"] = (all_df_copy < -20).astype(int).sum(axis=1)
	all_df["noutliers_30"] = (all_df_copy < -30).astype(int).sum(axis=1)
	all_df["noutliers_50"] = (all_df_copy < -50).astype(int).sum(axis=1)
	all_df["noutliers_75"] = (all_df_copy < -75).astype(int).sum(axis=1)
	all_df["noutliers_100"] = (all_df_copy < -100).astype(int).sum(axis=1)
	all_df["noutliers_200"] = (all_df_copy < -200).astype(int).sum(axis=1)

	all_df["outliers_02"] = (abs(all_df_copy) > 0.2).astype(int).sum(axis=1)
	all_df["outliers_03"] = (abs(all_df_copy) > 0.3).astype(int).sum(axis=1)
	all_df["outliers_04"] = (abs(all_df_copy) > 0.4).astype(int).sum(axis=1)
	all_df["outliers_05"] = (abs(all_df_copy) > 0.5).astype(int).sum(axis=1)
	all_df["outliers_1"] = (abs(all_df_copy) > 1).astype(int).sum(axis=1)
	all_df["outliers_2"] = (abs(all_df_copy) > 2).astype(int).sum(axis=1)
	all_df["outliers_3"] = (abs(all_df_copy) > 3).astype(int).sum(axis=1)
	all_df["outliers_5"] = (abs(all_df_copy) > 5).astype(int).sum(axis=1)
	all_df["outliers_7"] = (abs(all_df_copy) > 7).astype(int).sum(axis=1)
	all_df["outliers_10"] = (abs(all_df_copy) > 10).astype(int).sum(axis=1)
	all_df["outliers_12"] = (abs(all_df_copy) > 12).astype(int).sum(axis=1)
	all_df["outliers_15"] = (abs(all_df_copy) > 15).astype(int).sum(axis=1)
	all_df["outliers_18"] = (abs(all_df_copy) > 18).astype(int).sum(axis=1)
	all_df["outliers_20"] = (abs(all_df_copy) > 20).astype(int).sum(axis=1)
	all_df["outliers_30"] = (abs(all_df_copy) > 30).astype(int).sum(axis=1)
	all_df["outliers_50"] = (abs(all_df_copy) > 50).astype(int).sum(axis=1)
	all_df["outliers_75"] = (abs(all_df_copy) > 75).astype(int).sum(axis=1)
	all_df["outliers_100"] = (abs(all_df_copy) > 100).astype(int).sum(axis=1)
	all_df["outliers_200"] = (abs(all_df_copy) > 200).astype(int).sum(axis=1)

	all_df["empty_predictor_sum"] = (all_df_copy == 0).astype(int).sum(axis=1)

	all_df["var15_x_var38"] = all_df["var15"] * all_df["var38"]
	all_df["var15_x_saldo_var30"] = all_df["var15"] * all_df["saldo_var30"]
	all_df["saldo_var30_x_var38"] = all_df["saldo_var30"] * all_df["var38"]
	all_df["saldo_var42_x_var38"] = all_df["saldo_var42"] * all_df["var38"]
	all_df["saldo_var30_x_saldo_var42"] = all_df["saldo_var42"] * all_df["saldo_var30"]
	all_df["saldo_var30_x_var38"] = all_df["saldo_var42"] * all_df["var15"]
	all_df["saldo_medio_var5_ult1_x_saldo_var42"] = all_df["saldo_medio_var5_ult1"] * all_df["saldo_var42"]
	all_df["saldo_medio_var5_ult1_x_saldo_var30"] = all_df["saldo_medio_var5_ult1"] * all_df["saldo_var30"]
	all_df["saldo_medio_var5_ult1_x_var38"] = all_df["saldo_medio_var5_ult1"] * all_df["var38"]
	all_df["saldo_medio_var5_ult1_x_var15"] = all_df["saldo_medio_var5_ult1"] * all_df["var15"]
	
	#remove constant columns
	print(all_df.shape)
	all_df = all_df.loc[:,all_df.apply(pd.Series.nunique) != 1]
	print(all_df.shape)
	
	x_train = all_df.iloc[:test_separation]
	x_test = all_df.iloc[test_separation:]
	
	print(x_train.head(10))
	######################## GRADIENT BOOSTING & FEATURE SELECTION ###################################################
	
	print ("\nFeature selection with gradient boosting")
	
	gb_selection = GradientBoostingClassifier(random_state=1,  min_samples_leaf = 100, max_features ='auto', subsample = 0.5, learning_rate=0.005, n_estimators=2500)
	gb_selection.fit(x_train, y_train)
	print ("Score on train data:")
	print(roc_auc_score(y_train, gb_selection.predict_proba(x_train)[:,1]))
	
	#test submission with all features
	pd.DataFrame({"ID": x_test_id, "TARGET":  gb_selection.predict_proba(x_test)[:,1]}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\Santander\\subs\\gb_all.csv',index=False)
	
	features_lb = sorted(zip(map(lambda x: round(x, 4), gb_selection.feature_importances_), x_train.columns), reverse=True, key=lambda pair: pair[0])
	columns = [features_lb[i][1] for i in range(0,150)] ## I get 120
		
	print(columns)
	x_train= x_train[columns]
	x_test = x_test[columns]	
	
	# We have x_train, y_train, x_test, x_test_id (for submission)
		
	############################# MODELS 1ST TIER ############################################
	print("\nCreating models ....")
	
	############################# MODELS 1ST TIER ############################################
	
	rfcg_scores, rfce_scores, gb1_scores, gb2_scores,ab1_scores ,ab2_scores = ([] for i in range(6))
	
	predictions_train = pd.DataFrame(index=x_train.index, columns = ['rfcg', 'rfce', 'gb1', 'gb2', 'ab1'])
	predictions_test = pd.DataFrame(index=x_test.index, columns = ['rfcg', 'rfce', 'gb1', 'gb2', 'ab1'])
	
	#Random Forest Gini
	rfcg = RandomForestClassifier(n_estimators = 2500, n_jobs=-1, random_state=20, min_samples_leaf= 25, criterion='gini')
	#Random Forest Entropy
	rfce = RandomForestClassifier(n_estimators = 2500, n_jobs=-1, random_state=27, min_samples_leaf= 25, criterion='entropy')
	#Gradient Boosting 1
	gbc1 = GradientBoostingClassifier(random_state=1,  min_samples_leaf = 100, max_features ='auto', subsample = 0.5, learning_rate=0.005, n_estimators=2500)
	#Gradient Boosting 2
	gbc2 = GradientBoostingClassifier(random_state=3, n_estimators = 2000, min_samples_leaf = 50, max_features ='auto', subsample=0.9, learning_rate=0.01)
	#Ada Boost 1
	abc1 = AdaBoostClassifier(n_estimators=700, random_state=3,learning_rate=0.001)
		
		
	skf = StratifiedKFold(y_train, n_folds=7, random_state=1)
	for train_index, test_index in skf:
		x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
		y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
				
		#Random Forest Gini
		rfcg.fit(x_train_cv, y_train_cv)
		prob_rfcg = rfcg.predict_proba(x_test_cv)[:,1]
		rfcg_scores.append(roc_auc_score(y_test_cv,prob_rfcg))
		predictions_train.loc[test_index,'rfcg'] = prob_rfcg
		
		#Random Forest Entropy
		rfce.fit(x_train_cv, y_train_cv)
		prob_rfce = rfce.predict_proba(x_test_cv)[:,1]
		rfce_scores.append(roc_auc_score(y_test_cv,prob_rfce))
		predictions_train.loc[test_index,'rfce'] = prob_rfce
		
		#Gradient Boosting 1
		gbc1.fit(x_train_cv, y_train_cv)
		prob_gb1 = gbc1.predict_proba(x_test_cv)[:,1]
		gb1_scores.append(roc_auc_score(y_test_cv,prob_gb1))
		predictions_train.loc[test_index,'gb1'] = prob_gb1
		
		#Gradient Boosting 2
		gbc2.fit(x_train_cv, y_train_cv)
		prob_gb2 = gbc2.predict_proba(x_test_cv)[:,1]
		gb2_scores.append(roc_auc_score(y_test_cv,prob_gb2))
		predictions_train.loc[test_index,'gb2'] = prob_gb2
		
		#Ada Boost 1
		abc1.fit(x_train_cv, y_train_cv)
		prob_ab1 = abc1.predict_proba(x_test_cv)[:,1]
		ab1_scores.append(roc_auc_score(y_test_cv,prob_ab1))
		predictions_train.loc[test_index,'ab1'] = prob_ab1
		
		
	
	#Fit models for whole training set and calcuate predicitons for test set
	print("\nFitting models on whole training set...\n")
		
	#Random Forest Gini
	rfcg.fit(x_train, y_train)
	prob_rfcg = rfcg.predict_proba(x_test)[:,1]
	predictions_test['rfcg'] = prob_rfcg
	#Random Forest Entropy
	rfce.fit(x_train, y_train)
	prob_rfce = rfce.predict_proba(x_test)[:,1]
	predictions_test['rfce'] = prob_rfce
	#Gradient Boosting 1
	gbc1.fit(x_train, y_train)
	prob_gb1 = gbc1.predict_proba(x_test)[:,1]
	predictions_test['gb1'] = prob_gb1
	#Gradient Boosting 2
	gbc2.fit(x_train, y_train)
	prob_gb2 = gbc2.predict_proba(x_test)[:,1]
	predictions_test['gb2'] = prob_gb2
	#Ada Boost 1
	abc1.fit(x_train, y_train)
	prob_ab1 = abc1.predict_proba(x_test)[:,1]
	predictions_test['ab1'] = prob_ab1
	
	
	#Results
	print ("\nResults or SCV4: \n")
	print ("RFG the mean is %s and the std is %s" %(np.mean(rfcg_scores), np.std(rfcg_scores)))
	print ("RFE the mean is %s and the std is %s" %(np.mean(rfce_scores), np.std(rfce_scores)))
	print ("GB1 the mean is %s and the std is %s" %(np.mean(gb1_scores), np.std(gb1_scores)))
	print ("GB2 the mean is %s and the std is %s" %(np.mean(gb2_scores), np.std(gb2_scores)))
	print ("AB1 the mean is %s and the std is %s" %(np.mean(ab1_scores), np.std(ab1_scores)))
	
	
	#Correlations of scores
	print("Correlation for train is:\n") 
	print((predictions_train.astype(float)).corr())
	print("Correlation for test is:\n") 
	print((predictions_test.astype(float)).corr())
	
	##################################### RESULTS
	
	############################# LG STACKING 2ND TIER ############################################
	
	print("\nTraining Ensamble Logistic retression ....")
	lge = LogisticRegression(random_state=23, fit_intercept=True)
	param_grid = {'C': [0.01,0.1,1,2,5,7],'class_weight':['auto',None]}
	lgef = grid_search.GridSearchCV(estimator = lge, param_grid = param_grid, cv = 5, n_jobs=-1, scoring='roc_auc', error_score=0, verbose=2) 
	lgef_fit = lgef.fit(predictions_train[['rfcg', 'rfce', 'gb1', 'gb2','ab1']], y_train)
	predictions_train_ensamble_lg= lgef_fit.predict_proba(predictions_train[['rfcg', 'rfce', 'gb1', 'gb2','ab1']])[:,1]
	
	print("Best parameters found by grid search:", lgef.best_params_)
	print("Best CV score:", lgef.best_score_)
		
	print ("score training ensamble lg: ")
	print(roc_auc_score(y_train, predictions_train_ensamble_lg))
	
	predictions_test_ensamble_lg= lgef_fit.predict_proba(predictions_test[['rfcg', 'rfce', 'gb1', 'gb2','ab1']])[:,1]
	pd.DataFrame({"ID": x_test_id, "TARGET":  predictions_test_ensamble_lg}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\Santander\\subs\\ensamble_lg.csv',index=False)
	
	############################# RF STACKING 2ND TIER ############################################
	
	print("\nTraining Ensamble Random Forest ....")
	rffc = RandomForestClassifier(n_estimators = 2000, n_jobs=-1, random_state=20, min_samples_leaf= 25, criterion='gini')
	param_grid = {'min_samples_leaf': [300,250]}
	rffc_gs = grid_search.GridSearchCV(estimator = rffc, param_grid = param_grid, cv = 4, n_jobs=-1, scoring='roc_auc', error_score=0, verbose=2) 
	rffc_gs_fit = rffc_gs.fit(predictions_train[['rfcg', 'rfce', 'gb1', 'gb2','ab1']], y_train)
	predictions_train_ensamble_rf= rffc_gs_fit.predict_proba(predictions_train[['rfcg', 'rfce', 'gb1', 'gb2','ab1']])[:,1]
	
	print("Best parameters found by grid search:", rffc_gs.best_params_)
	print("Best CV score:", rffc_gs.best_score_)
			
	print ("score training ensamble rf: ")
	print(roc_auc_score(y_train, predictions_train_ensamble_rf))
		
	predictions_test_ensamble_rf= rffc_gs_fit.predict_proba(predictions_test[['rfcg', 'rfce', 'gb1', 'gb2','ab1']])[:,1]
	pd.DataFrame({"ID": x_test_id, "TARGET":  predictions_test_ensamble_rf}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\Santander\\subs\\ensamble_rf.csv',index=False)
	
	
	############################# MY WEIGHTED AVG STACKING 2ND TIER ############################################

	print(" my weights")
	
	max_score = 0
	for w_rfg in range(0,5):
		for w_rfe in range(0,5):
			for w_gb1 in range(2,5):
				for w_gb2 in range(0,5):
					for w_ab1 in range(0,5):
						prob_my = ((predictions_train['rfcg']*w_rfg)+(predictions_train['rfce']*w_rfe)+(predictions_train['gb1']*w_gb1)+(predictions_train['gb2']*w_gb2)+(predictions_train['ab1']*w_ab1)) / (w_svm+w_lglda+w_rfg+w_knn+w_gb1+w_ab1+w_gb2)
						prob_my = prob_my.astype(np.float64)
						myscore = roc_auc_score(y_train, prob_my)
						if myscore > max_score:
							final_rfg = w_rfg
							final_rfe = w_rfe
							final_gb1 = w_gb1
							final_gb2 = w_gb2
							final_ab1 = w_ab1
							max_score = myscore
							
	print ("final weights:")
	print ("w_rfg: %s, w_rfe: %s, w_gb1: %s, w_gb2: %s, w_ab1: %s" %(final_rfg, final_rfe, final_gb1, final_gb2, final_ab1))
	prob_my = ((predictions_train['rfcg']*final_rfg)+(predictions_train['rfce']*final_rfe)+(predictions_train['gb1']*final_gb1)+(predictions_train['gb2']*final_gb2)+(predictions_train['ab1']*final_ab1)) / (final_rfg+final_rfe+final_gb1+final_gb2+final_ab1)
	prob_my = prob_my.astype(np.float64)
	print("score of my ensamble")
	print(roc_auc_score(y_train, prob_my))
	
	#My ensamble submission
	prob_my_test = ((predictions_test['rfcg']*final_rfg)+(predictions_test['rfce']*final_rfe)+(predictions_test['gb1']*final_gb1)+(predictions_test['gb2']*final_gb2)+(predictions_test['ab1']*final_ab1)) / (final_rfg+final_rfe+final_gb1+final_gb2+final_ab1)
	pd.DataFrame({"ID": x_test_id, "TARGET":  prob_my_test}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\Santander\\subs\\ensamble_my.csv',index=False)

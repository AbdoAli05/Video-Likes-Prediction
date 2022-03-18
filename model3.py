from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from preProcessing import get_data_set


x_train, x_test, y_train, y_test = get_data_set()
# Random Forest
# Hypermeter Turning



nEstimator = [100]
depth = [20]
# [10, 15, 20, 25, 30]
RF = RandomForestRegressor()
hyperParam = [{'n_estimators': nEstimator, 'max_depth': depth}]
# 5
gsv = GridSearchCV(RF, hyperParam, cv=2, verbose=1, scoring='r2', n_jobs=-1)

gsv.fit(x_train, y_train)
print("Best HyperParameter: ", gsv.best_params_)
print(gsv.best_score_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator), len(depth))

plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth = gsv.best_params_['max_depth']
nEstimators = gsv.best_params_['n_estimators']

print('max_depth', 'n_estimators')
print(maxDepth, nEstimators)
# Random Forest using the optimal hypermeter


model = RandomForestRegressor(n_estimators=nEstimators, max_depth=maxDepth)
model.fit(x_train, y_train)

# predicting the  test set results
y_pred = model.predict(x_test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :", model.score(x_test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data=d1)
# print(SK)

lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data=SK, size=10)
fig1 = lm1.fig
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale=1.5)

plt.show()

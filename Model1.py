# Loading library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from preProcessing import get_data_set

warnings.filterwarnings('ignore')
# # REGRESSION ANALYSIS

# LINEAR REGRESSION
x_train, x_test, y_train, y_test = get_data_set()


model = LinearRegression()
model.fit(x_train, y_train)

# predicting the  test set results
y_pred = model.predict(x_test)
print('Root means score', round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print('Accuracy : ', round(r2_score(y_test, y_pred), 2))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data=d1)
# print(SK)

lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data=SK, size=10)
fig1 = lm1.fig
fig1.suptitle("Sklearn_Multy_Liar ", fontsize=18)
sns.set(font_scale=1.5)

plt.show()



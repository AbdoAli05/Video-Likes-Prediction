from sklearn import neighbors
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from preProcessing import get_data_set

scaling = MinMaxScaler()
x_train, x_test, y_train, y_test = get_data_set()

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(x_train)
X_test_poly = poly.fit_transform(x_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(X_test_poly)

print("Poly Score : ", round(poly_model.score(X_test_poly, y_test), 2)*100)

X_train_scaled = scaling.fit_transform(X_train_poly)
X_test_scaled = scaling.transform(X_test_poly)

# lasso & Ridge With Org Feature

lasso = Lasso(alpha=0.01, max_iter=100)
lasso.fit(X_train_poly, y_train)
y_pred = lasso.predict(X_test_poly)
Lasso_R2_test_score = r2_score(y_test, y_pred)

ridge = Ridge(alpha=0.01, normalize=True)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
Ridge_test_score = r2_score(y_test, y_pred)


# lasso & Ridge With Scaling Feature

lasso = Lasso(alpha=0.01, max_iter=100)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
Lasso_R2_test_score_scaled = r2_score(y_test, y_pred)

ridgereg = Ridge(alpha=0.01, normalize=True)
ridgereg.fit(X_train_scaled, y_train)
y_pred = ridgereg.predict(X_test_scaled)
Ridge_test_score_scaled = r2_score(y_test, y_pred)

print("Lasso Score : ", round(Lasso_R2_test_score, 2)*100)
print("Lasso Score With Scaling : ", round(Lasso_R2_test_score_scaled, 2)*100)
print("Ridge Score : ", round(Ridge_test_score, 2)*100)
print("Ridge Score With Scaling ", round(Ridge_test_score_scaled, 2)*100)

print("Score KNeighborsClassifier")
for k in range(1, 30):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("\tScore With", k, "Neighbors", round(r2_score(y_test, y_pred) * 100, 2))

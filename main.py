import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def RegresionLineal():
    dataset = pd.read_csv('weatherHistory.csv')

    dataset.drop(dataset.columns[[0, 1, 2, 4, 7, 8, 9, 11]], axis=1, inplace=True)
    X = dataset.iloc[:, 0].values
    y1 = dataset.iloc[:, 1].values
    y2 = dataset.iloc[:, 2].values
    y3 = dataset.iloc[:, 3].values

    plt.scatter(X, y1, color="red")
    plt.title("Temperature (C) vs Humidity")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y1, test_size=1 / 3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title("Temperature (C) vs Humidity (Regresion)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity")
    plt.show()
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title("Temperature (C) vs Humidity (Test)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity")
    plt.show()
    r2 = r2_score(y_test, y_pred)
    print('r2 score Temperature (C) vs Humidity', r2)

    plt.scatter(X, y2, color="blue")
    plt.title("Temperature (C) vs Wind Speed (km/h)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Wind Speed (km/h)")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y2, test_size=1 / 3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    plt.scatter(X_train, y_train, color="blue")
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title("Temperature (C) vs Wind Speed (km/h) (Regresion)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Wind Speed (km/h)")
    plt.show()
    plt.scatter(X_test, y_test, color="blue")
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title("Temperature (C) vs Wind Speed (km/h) (Test)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Wind Speed (km/h)")
    plt.show()
    r2 = r2_score(y_test, y_pred)
    print('r2 score Temperature (C) vs Wind Speed (km/h)', r2)

    plt.scatter(X, y3, color="green")
    plt.title("Temperature (C) vs Pressure (millibars)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Pressure (millibars)")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y3, test_size=1 / 3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    plt.scatter(X_train, y_train, color="green")
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title("Temperature (C) vs Pressure (millibars) (Regresion)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Pressure (millibars)")
    plt.show()
    plt.scatter(X_test, y_test, color="green")
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title("Temperature (C) vs Pressure (millibars) (Test)")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Pressure (millibars)")
    plt.show()
    r2 = r2_score(y_test, y_pred)
    print('r2 score Temperature (C) vs Pressure (millibars)', r2)


def RegresionLinealMultivariable():
    dataset = pd.read_csv('weatherHistory.csv')
    dataset.drop(dataset.columns[[0, 2, 4, 7, 8, 9, 11]], axis=1, inplace=True)
    dataset['Summary'] = dataset['Summary'].astype('category')
    dataset['Summary'] = dataset['Summary'].cat.codes
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    r2 = r2_score(y_test, y_pred)
    print('r2 score Multivariable lineal', r2)


def RegresionPolinomial():
    dataset = pd.read_csv('weatherHistory.csv')

    dataset.drop(dataset.columns[[0, 1, 2, 4, 7, 8, 9, 11]], axis=1, inplace=True)
    X = dataset[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']].values
    y = dataset['Pressure (millibars)'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    new_data = np.array([[9, 7, 0.9]])  # Ejemplo de valores
    new_data_poly = poly.transform(new_data)
    predicted = model.predict(new_data_poly)
    print(predicted)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset.iloc[:, 1], dataset.iloc[:, 2], dataset.iloc[:, 3], color='blue')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Humidity')
    ax.set_zlabel('Wind Speed (km/h)')
    plt.show()


def main():
    RegresionLineal()
    print("______________________")
    RegresionLinealMultivariable()
    print("______________________")
    RegresionPolinomial()


if __name__ == "__main__":
    main()

import csv
import sys
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(str(row[0].split('-')[0])+str(row[0].split('-')[1])+str(row[0].split('-')[2])))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates_train = np.reshape(dates[:len(dates)/2], (len(dates)/2,1))
    prices_train = prices[:len(prices)/2]
    dates = np.reshape(dates,(len(dates),1))
    print(len(dates), "datos en total")
    print("Preparando entrenamiento")
    #svr_len = SVR(kernel = 'linear', C=1e3)
    #svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)
    print("Entrenando... ")
    #svr_len.fit(dates, prices)
    #svr_poly.fit(dates, prices)
    svr_rbf.fit(dates_train, prices_train)
    print("Entrenamiento completado, mostrando resultados... ")
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    #plt.plot(dates, svr_len.predict(dates), color='green', label='Lineal model')
    #plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Vector de regression para soporte')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_len.predict(x)[0], svr_poly.predict(x)[0]

get_data(sys.argv[1])

predicted_price = predict_prices(range(len(dates)), prices,29)

print(predicted_price)

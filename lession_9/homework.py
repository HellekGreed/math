import numpy as np
import matplotlib.pyplot as plt

print('Даны значения величины заработной платы заемщиков банка (zp) и значения их поведенческого кредитного скоринга (ks):')
print('zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]')
print('Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая переменная).')
print('Произвести расчет как с использованием intercept, так и без.')
print()
print('C intercept')
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

n = len(ks)
b1 = (np.mean(zp*ks)-np.mean(zp)*np.mean(ks))/(np.mean(zp**2) - np.mean(zp)**2)
print(round(b1, 2))

b0 = np.mean(ks)-b1*np.mean(zp)
print(round(b0, 2))

y_pred = b0 + b1 * zp
print(y_pred)

plt.scatter(zp, ks)
plt.plot(zp, y_pred)
plt.show()

mse = np.sum(((b0 + b1 * zp) - ks) ** 2 / n)
print(mse)

mse_ = ((ks - y_pred)**2).sum() / n
print(mse_)
print()

print('Без intercept')

zp1 = zp.reshape(1, n)
ks1 = ks.reshape(1, n)

b1 = np.dot(np.dot(np.linalg.inv(np.dot(zp1, zp1.T)), zp1), ks1.T)[0][0]
print(round(b1, 2))

y_pred1 = b1 * zp
print(y_pred1)
plt.scatter(zp, ks)
plt.plot(zp, y_pred, 'b', label = 'с intercept')
plt.plot(zp, y_pred1, 'r', label = 'без interсept')
plt.show()

print()
print('Посчитать коэффициент линейной регрессии при заработной плате (zp), используя градиентный спуск (без intercept).')
print()

alpha = 1e-6
b1 = 0.1

def mse_(b1, y=ks, X=zp, n=10):
    return np.sum((b1 * X - y) ** 2) / n

for i in range(1000):
    fp = (1 / n) * np.sum(2 * (b1 * zp - ks) * zp)
    b1 -= alpha * fp
    if i % 100 == 0:
        print(f'Итерация: {i}, b1 : {b1}, mse: {mse_(b1) }')

y_pred2 = b1 * zp
print(y_pred2)
print()

plt.scatter(zp, ks)
plt.plot(zp, y_pred, 'b:', label = 'с интерсептом')
plt.plot(zp, y_pred2, 'r:', label = 'без интерсепта')
plt.show()        

print()
print('Произвести вычисления как в пункте 2, но с вычислением intercept. Учесть, что изменение коэффициентов должно производиться на каждом шаге одновременно')
print('(то есть изменение одного коэффициента не должно влиять на изменение другого во время одной итерации).')
print()

alpha = 5e-5

b0 = 0.1
b1 = 0.1

def mse_(b0, b1, y = ks, X = zp, n = 10):
    return np.sum((b0 + b1 * X - y) ** 2) / n

for i in range(1000000):
    y_pred3 = b0 + b1 * zp
    b0 -=alpha * (2 / n) * np.sum((y_pred3 - ks))
    b1 -=alpha * (2 / n) * np.sum((y_pred3 - ks)*zp)
    if i % 100000 == 0:
        print(f"Итерация: {i}, b1 : {b1}, b0 : {b0}, mse: {mse_(b0,b1)}")

from scipy import stats
import numpy as np

print('Даны две  независимые выборки. Не соблюдается условие нормальности')
print('x1  380,420, 290')
print('y1 140,360,200,900')
print()

x1 = np.array([380, 420, 290])
y1 = np.array([140, 360, 200, 900])

print(stats.mannwhitneyu(x1, y1))

print('p-value больше 0.05, статистических различий нет.')

print()
print('Исследовалось влияние препарата на уровень давления пациентов. Сначала измерялось давление до приема препарата, потом через 10 минут и через 30 минут.') 
print('Есть ли статистически значимые различия?')
print('1е измерение до приема препарата: 150, 160, 165, 145, 155')
print('2е измерение через 10 минут: 140, 155, 150,  130, 135')
print('3е измерение через 30 минут: 130, 130, 120, 130, 125')
print()

before = np.array([150, 160, 165, 145, 155])
after_20 = np.array([140, 155, 150, 130, 135])
after_30 = np.array([130, 130, 120, 130, 125])

print(stats.friedmanchisquare(before, after_20, after_30))

print('p-value меньше 0.05, статистические различия есть.')

print()
print('Сравните 1 и 2 е измерения, предполагая, что 3го измерения через 30 минут не было.')
print()
print(stats.wilcoxon(before, after_20))
print('p-value больше 0.05, статистических различий нет.')
print()
print('Даны 3 группы  учеников плавания.')
print('В 1 группе время на дистанцию 50 м составляют: 56, 60, 62, 55, 71, 67, 59, 58, 64, 67')
print('Вторая группа : 57, 58, 69, 48, 72, 70, 68, 71, 50, 53')
print('Третья группа: 57, 67, 49, 48, 47, 55, 66, 51, 54')
print()

group_one = np.array([56, 60, 62, 55, 71, 67, 59, 58, 64, 67])
group_two = np.array([57, 58, 69, 48, 72, 70, 68, 71, 50, 53])
group_three = np.array([57, 67, 49, 48, 47, 55, 66, 51, 54])

print(stats.kruskal(group_one, group_two, group_three))

print('p-value больше 0.05, статистических различий нет.')

print()
print('Заявляется, что партия изготавливается со средним арифметическим 2,5 см.') 
print('Проверить данную гипотезу, если известно, что размеры изделий подчинены нормальному закону распределения.')
print('Объем выборки 10, уровень статистической значимости 5%')
print('2.51, 2.35, 2.74, 2.56, 2.40, 2.36, 2.65, 2.7, 2.67, 2.34')
print()

x = np.array([2.51, 2.35, 2.74, 2.56, 2.40, 2.36, 2.65, 2.7, 2.67, 2.34])
std = (((x - x.mean())**2).sum() / (10 - 1))**0.5
tn = (x.mean() - 2.5) / (std / 10**0.5) 
t = abs(stats.t.ppf(0.05, 9))

print(f't = {t}, Tn = {tn}')
print('t > Tn, гипотеза верна.')
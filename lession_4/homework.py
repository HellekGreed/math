from scipy import stats

# Случайная непрерывная величина A имеет равномерное распределение на промежутке (200, 800].
# Найдите ее среднее значение и дисперсию.
mid_val = (200 + 800) / 2
dis = (800 - 200) ** 2 / 12
print(mid_val)
print(dis)

# О случайной непрерывной равномерно распределенной величине B известно, что ее дисперсия равна 0.2. 
# Можно ли найти правую границу величины B и ее среднее значение зная, что левая граница равна 0.5? Если да, найдите ее.

rigth_bord = (0.2 * 12)**0.5 + 0.5
print(round(rigth_bord, 3))

#Непрерывная случайная величина X распределена нормально и задана плотностью распределения
# f(x) = (1 / (4 * sqrt(2pi))) * exp((-(x+2)**2) / 32)
# Найдите: а). M(X) б). D(X) в). std(X) (среднее квадратичное отклонение)

M = -2
D = 16
std = 4
print(M)
print(D)
print(std)

# Рост взрослого населения города X имеет нормальное распределение.
# Причем, средний рост равен 174 см, а среднее квадратичное отклонение равно 8 см.

M = 174
std = 8

# Какова вероятность того, что случайным образом выбранный взрослый человек имеет рост:
# а). больше 182 см
print(round(1 - stats.norm(M,std).cdf(182), 4))
Z = (182 - 174) / 8
print(round(1 - 0.8413, 4))
# б). больше 190 см
print(round(1 - stats.norm(M,std).cdf(190), 4))
# в). от 166 см до 190 см
print(round(stats.norm(M,std).cdf(190) - stats.norm(M,std).cdf(166), 4))
# г). от 166 см до 182 см
print(round(stats.norm(M,std).cdf(182) - stats.norm(M,std).cdf(166), 4))
# д). от 158 см до 190 см
print(round(stats.norm(M,std).cdf(190) - stats.norm(M,std).cdf(158), 4))
# е). не выше 150 см или не ниже 190 см
print(round(stats.norm(M,std).cdf(150) + 1 - stats.norm(M,std).cdf(190), 4))
# ё). не выше 150 см или не ниже 198 см
print(round(stats.norm(M,std).cdf(150) + 1 - stats.norm(M,std).cdf(198), 4))
# ж). ниже 166 см.
print(round(stats.norm(M,std).cdf(166), 4))


# На сколько сигм (средних квадратичных отклонений) отклоняется рост человека, 
# равный 190 см, от математического ожидания роста в популяции, в которой M(X) = 178 см и D(X) = 25 кв.см?

sig = (190 - 178) / 25**0.5
print(sig)



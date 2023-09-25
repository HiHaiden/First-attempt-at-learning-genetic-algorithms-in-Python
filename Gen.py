from deap import base, algorithms
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# константы задачи
BOUND_LOW, BOUND_UP = -20.0, -2.3  # границы
EPS = 0.001  # точность

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 100  # максимальное количество поколений
HALL_OF_FAME_SIZE = 5  # Зал славы
RANDOM_SEED = 42 # зерно для генератора случайных чисел
random.seed(RANDOM_SEED)

# функция вычисления степени 2 целого числа
def POWER_OF_TWO(n):
    sqr = 1
    r = 0
    while sqr <= n:
        sqr = sqr * 2
        r += 1
    return r

# вычисление необходимой длины особи
LENGTH = POWER_OF_TWO((BOUND_UP - BOUND_LOW) / EPS)  # длина подлежащей оптимизации битовой строки

# создание класса для описания значения приспособленности особей
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# создание класса для представления каждого индивидуума
creator.create("Individual", list, fitness=creator.FitnessMax)

# функция расчёта приспособленности
def func(individual):
    global f
    x = int(str(individual).replace(',', '').replace('[', '').replace(']', '').replace(' ', ''), 2)
    h = (BOUND_UP - BOUND_LOW) / (2 ** LENGTH - 1)
    xx = BOUND_LOW + x * h
    f = (np.cos(2*xx) / (xx*xx))
    return f,  # вернуть кортеж

# экземпляр класса base.Toolbox
toolbox = base.Toolbox()
#определение функции для генерации случайных значений
toolbox.register("zeroOrOne", random.randint, 0, 1)
#определение функции для генерации отдельного индивидуума
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, LENGTH)
#определение функции для создания начальной популяции
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
population = toolbox.populationCreator(n=POPULATION_SIZE)

#вычисление приспособленности каждой особи на основе func
toolbox.register("evaluate", func)
#отбор особей (турнирный с размером 3)
toolbox.register("select", tools.selTournament, tournsize=3)
#скрещивание особей (одноточечный кроссовер)
toolbox.register("mate", tools.cxOnePoint)
#мутация инвертированием бита с вероятностью indpb
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / LENGTH)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          verbose=True)

def convert(list):
    r = int(str(list).replace(',', '').replace('[', '').replace(']', '').replace(' ', ''), 2)
    h = (BOUND_UP - BOUND_LOW) / (2 ** LENGTH - 1)
    res = BOUND_LOW + r * h
    return res

# вывод лучших результатов:
print("-- Индивидуумы в зале славы = ")
for i in range(HALL_OF_FAME_SIZE):
    print(convert(hof.items[i]), sep="\n")
print("-- Лучший индивидуум = ", convert(hof.items[0]))
print("-- Лучшая приспособленность = ", hof.items[0].fitness.values[0])

# извлечение статистик:
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

# график статистик:
sns.set_style("whitegrid")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show() 
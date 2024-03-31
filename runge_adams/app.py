import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Функция правой части уравнения y' = ((x-1)*y)/(x^2)
def f(x, y):
    return ((x - 1) * y) / (x**2)

# Точное решение уравнения
def exact_solution(x):
    return x * np.exp(1/x)

# Метод Рунге-Кутта второго порядка
def runge_kutta_2(x0, y0, h, end_x):
    x_values = [x0]
    y_values = [y0]
    
    while x_values[-1] < end_x:
        x_i = x_values[-1]
        y_i = y_values[-1]
        
        k1 = h * f(x_i, y_i)
        k2 = h * f(x_i + 0.5*h, y_i + 0.5*k1)
        
        y_next = y_i + k2
        x_values.append(x_i + h)
        y_values.append(y_next)
    
    return x_values, y_values

# Метод Адамса второго порядка
def adams_2(x_values, y_values, h, end_x):
    while x_values[-1] < end_x:
        x_n = x_values[-1]
        y_n = y_values[-1]

        # Прогноз
        y_predict = y_n + h * (3/2 * f(x_n, y_n) - 1/2 * f(x_n - h, y_values[-2]))

        # Коррекция
        y_next = y_n + h * (1/2 * f(x_n + h, y_predict) + 1/2 * f(x_n, y_n))
        
        x_values.append(x_n + h)
        y_values.append(y_next)
    
    return x_values, y_values

# Вывод результатов в консоль
def print_results(x_values1, y_values1, y_values2, exact_values, error_values):
    print("{:<10} {:<25} {:<25} {:<25} {:<25}".format(
        'Отрезок', 'Приблизительное (h=0.1)', 
        'Уточненное (h=0.1)', 'Точное решение', 'Погрешность'))
    
    for x, y1, y2, exact, error in zip(x_values1, y_values1, y_values2, exact_values, error_values):
        print("{:<10.5f} {:<25.5f} {:<25.5f} {:<25.5f} {:<25.5f}".format(x, y1, y2, exact, error))



# Вывод результатов погрешности метода Рунге в консоль
def print_results_runge(x_values, y_values, exact_values, error_values):
    print("{:<10} {:<25} {:<25} {:<25}".format(
        'Отрезок', 'Уточненное (h=0.2)', 'Точное решение', 'Погрешность'))
    
    for x, y, exact, error in zip(x_values, y_values, exact_values, error_values):
        print("{:<10.5f} {:<25.5f} {:<25.5f} {:<25.5f}".format(x, y, exact, error))


if __name__ == "__main__":
    # Начальные условия
    x0 = 1
    y0 = np.exp(1)
    h1 = 0.1
    h2 = 0.2
    end_x = 2

    # Решение первыми 5 точками методом Рунге-Кутта
    x_values1, y_values1 = runge_kutta_2(x0, y0, h1, x0 + 5*h1)
    
    # Уточнение решения с шагом h=0.1 с использованием метода Адамса
    x_values_adams, y_values_adams = adams_2(x_values1, y_values1, h1, end_x)
    
    # Вычисление точного решения для сравнения
    exact_values = [exact_solution(x) for x in x_values1]

    # Вычисление погрешности для Адамса с шагом h=0.1
    error_values_adams = [abs(y1 - y2) for y1, y2 in zip(y_values1, y_values_adams)]

    # Вывод результатов в консоль
    print_results(x_values1, y_values1, y_values_adams, exact_values, error_values_adams)

    # Уточнение решения методом Рунге с шагом h=0.2
    x_values_runge, y_values_runge = runge_kutta_2(x0, y0, h2, end_x)
    
    # Вычисление погрешности для Рунге с шагом h=0.2
    error_values_runge = [abs(y1 - y2) for y1, y2 in zip(y_values_runge, exact_values)]

    print('\n'*2)

    print_results_runge(x_values_runge, y_values_runge, exact_values, error_values_runge)

    # Создание массивов данных для графиков
    x_values_rk = x_values1  # первые 5 точек + начальная точка
    y_values_rk = y_values1  # соответствующие значения y

    x_values_adams = x_values1[-5:] + x_values_adams  # оставшиеся точки + точки Адамса
    y_values_adams = y_values1[-5:] + y_values_adams  # соответствующие значения y

    # Построение графиков
    plt.figure(figsize=(12, 8))
    plt.plot(x_values_rk, y_values_rk, label='Приблизительное решение (Рунге)')
    plt.plot(x_values_adams, y_values_adams, label='Уточненное решение (Адамс)', linestyle='--')
    plt.plot(x_values_runge, y_values_runge, label='Уточненное решение (Рунге h=0.2)', linestyle='-.')
    plt.plot(x_values1, exact_values, label='Точное решение', linestyle='-')
    plt.title('Решение задачи Коши для ОДУ с учетом погрешности')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
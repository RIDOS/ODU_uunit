import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Функция правой части уравнения y' = ((x-1)*y)/(x^2)
def f(x, y):
    return ((x - 1) * y) / (x**2)

# Точное решение уравнения
def exact_solution(x):
    return x * np.exp(1/x)

# Явный метод Эйлера
def euler_method(x0, y0, h, end_x):
    x_values = [x0]
    y_values = [y0]
    
    while x_values[-1] < end_x:
        x_i = x_values[-1]
        y_i = y_values[-1]
        
        y_next = y_i + h * f(x_i, y_i)
        
        x_values.append(x_i + h)
        y_values.append(y_next)
    
    return x_values, y_values

# Исправленный метод Эйлера
def corrected_euler_method(x0, y0, h, end_x):
    x_values = [x0]
    y_values = [y0]
    
    while x_values[-1] < end_x:
        x_i = x_values[-1]
        y_i = y_values[-1]
        
        y_half = y_i + 0.5 * h * f(x_i, y_i)
        y_next = y_i + h * f(x_i + 0.5 * h, y_half)
        
        x_values.append(x_i + h)
        y_values.append(y_next)
    
    return x_values, y_values

# Метод Рунге для уточнения
def runge_method(y1, y2, p):
    return y1 + (y1 - y2) / ((2**p) - 1)

# Вывод результатов в консоль
def print_results(x_values1, y_values1, y_values2, y_values_runge, exact_values):
    print("{:<10} {:<25} {:<25} {:<25} {:<25} {:<25}".format(
        'Отрезок', 'Приблизительное Эйлера', 
        'Уточненное Эйлера', 'Уточненное методом Рунге', 
        'Точное решение', 'Погрешность'))
    
    for x, y1, y2, y_runge, exact in zip(x_values1, y_values1, y_values2, y_values_runge, exact_values):
        error = abs(y_runge - exact)
        print("{:<10.5f} {:<25.5f} {:<25.5f} {:<25.5f} {:<25.5f} {:<25.5f}".format(x, y1, y2, y_runge, exact, error))



if __name__ == "__main__":
    # Начальная точка отрезка
    x0 = 1
    # Начальное приближение
    y0 = np.exp(1)
    # Шаг
    h = 0.1
    # Конец отрезка
    end_x = 2

    # Явный метод Эйлера
    x_values_euler, y_values_euler = euler_method(x0, y0, h, end_x)
    
    # Исправленный метод Эйлера
    x_values_corrected, y_values_corrected = corrected_euler_method(x0, y0, h, end_x)
    
    # Вычисление точного решения
    exact_values = [exact_solution(x) for x in x_values_euler]

    # Уточнение по методу Рунге
    y_values_runge = [runge_method(y_values_euler[i], y_values_corrected[i], 2) for i in range(len(y_values_euler))]

    # Вывод результатов в консоль
    print_results(x_values_corrected, y_values_euler, y_values_corrected, y_values_runge, exact_values)

    # Построение графиков
    plt.figure(figsize=(12, 8))
    plt.plot(x_values_euler, y_values_euler, label='Приблизительное решение (Эйлер)')
    plt.plot(x_values_corrected, y_values_corrected, label='Приблизительное решение (Исправленный Эйлер)')
    plt.plot(x_values_euler, y_values_runge, label='Уточненное решение (метод Рунге)')
    plt.plot(x_values_euler, exact_values, label='Точное решение', linestyle='--')
    plt.title('Решение задачи Коши для ОДУ')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

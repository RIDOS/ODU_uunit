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

if __name__ == '__main__':
    # Задаем начальные значения
    x0 = 1.0
    y0 = np.exp(1.0)
    end_x = 2.0
    h = 0.1

    # Вычисляем первые пять точек методом Рунге-Кутта
    x_values_rk, y_values_rk = runge_kutta_2(x0, y0, h, x0 + 0.5*h*4)

    # Вычисляем остальные точки методом Адамса
    x_values_adams, y_values_adams = adams_2(x_values_rk[-5:], y_values_rk[-5:], h, end_x)

    # Объединяем результаты
    x_values_combined = x_values_rk + x_values_adams
    y_values_combined = y_values_rk + y_values_adams

    # Вычисляем точное решение
    x_exact = np.linspace(x0, end_x, 100)
    y_exact = exact_solution(x_exact)

    # Создаем таблицу с результатами
    df = pd.DataFrame({
        'x': x_values_combined,
        'Комбинированный метод': y_values_combined,
        'Точное решение': [exact_solution(x) for x in x_values_combined]
    })
    print(df)

    # Строим графики
    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, label='Точное решение', color='blue')
    plt.plot(x_values_combined, y_values_combined, label='Комбинированный метод', linestyle='--', marker='o', color='red')
    plt.title('Решение задачи Коши методами Рунге-Кутта и Адамса')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

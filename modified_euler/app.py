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

# Метод Рунге-Кутта четвёртого порядка
def runge_kutta_method(x0, y0, h, end_x):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < end_x:
        x_i = x_values[-1]
        y_i = y_values[-1]

        k1 = h * f(x_i, y_i)
        k2 = h * f(x_i + 0.5 * h, y_i + 0.5 * k1)
        k3 = h * f(x_i + 0.5 * h, y_i + 0.5 * k2)
        k4 = h * f(x_i + h, y_i + k3)

        y_next = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6

        x_values.append(x_i + h)
        y_values.append(y_next)

    return x_values, y_values

if __name__ == "__main__":
    # Начальные параметры
    x0 = 1
    y0 = np.exp(1)
    end_x = 2

    # Выполнение методов
    x_values_euler, y_values_euler = euler_method(x0, y0, 0.1, end_x)
    x_values_corrected_euler, y_values_corrected_euler = corrected_euler_method(x0, y0, 0.1, end_x)
    x_values_runge_kutta, y_values_runge_kutta = runge_kutta_method(x0, y0, 0.2, end_x)
    exact_values = [exact_solution(x) for x in x_values_runge_kutta]

    # Создание DataFrame
    df = pd.DataFrame({
        'x': x_values_runge_kutta,
        'Явный Эйлер (h=0.1)': np.interp(x_values_runge_kutta, x_values_euler, y_values_euler),
        'Исправленный Эйлер (h=0.1)': np.interp(x_values_runge_kutta, x_values_corrected_euler, y_values_corrected_euler),
        'Рунге-Кутта (h=0.2)': y_values_runge_kutta,
        'Точное решение': exact_values
    })

    # Снова создаем DataFrame с результатами
    print(df)

    # Построение графиков
    plt.figure(figsize=(12, 8))
    plt.plot(df['x'], df['Явный Эйлер (h=0.1)'], label='Явный метод Эйлера (h=0.1)')
    plt.plot(df['x'], df['Исправленный Эйлер (h=0.1)'], label='Исправленный Эйлер (h=0.1)')
    plt.plot(df['x'], df['Рунге-Кутта (h=0.2)'], label='Метод Рунге-Кутта (h=0.2)')
    plt.plot(df['x'], df['Точное решение'], label='Точное решение', linestyle='--')
    plt.title('Решение задачи Коши для ОДУ 1 порядка.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
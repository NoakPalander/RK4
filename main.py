#!venv/bin/python

from matplotlib import pyplot as plt
from matplotlib import gridspec


def euler_method(initial: tuple, delta_x: float, target: float, derv) -> list:
    assert(delta_x != 0)
    x, y = initial
    steps = int(target / delta_x)

    points = [initial]
    for _ in range(steps):
        y += delta_x * derv(x, y)
        x += delta_x
        points.append((x, y))

    return points


def euler_deluxe_method(initial: tuple, delta_x: float, target: float, derv) -> list:
    assert(delta_x != 0)
    x, y = initial
    steps = int(target / delta_x)

    points = [initial]
    for _ in range(steps):
        y2 = y + derv(x, y) * delta_x
        x2 = x + delta_x

        y += delta_x * (derv(x, y) + derv(x2, y2)) / 2
        x += delta_x
        points.append((x, y))

    return points


def runge_kutta(initial: tuple, delta_x: float, target: float, derv) -> list:
    assert(delta_x != 0)
    x, y = initial
    steps = int(target / delta_x)

    points = [initial]
    for _ in range(steps):
        k1 = derv(x, y)
        k2 = derv(x + delta_x / 2, y + (delta_x / 2) * k1)
        k3 = derv(x + delta_x / 2, y + (delta_x / 2) * k2)
        k4 = derv(x + delta_x, y + delta_x * k3)

        y += (delta_x / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += delta_x
        points.append((x, y))

    return points


def main():
    gs = gridspec.GridSpec(2, 2)
    
    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title(r'$\Delta x = 0.2$')
    ax1.set_xlim([0.4, 1])
    ax1.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label=r"$y_e$: Euler's method", color='red')
    
    ax1.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label=r"$y_{ei}$: Euler's improved method", color='blue')
    
    ax1.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label=r"$y_{rk}$: Runge Kutta 4th order", color='green')
    
    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title(r'$\Delta x = 0.1$')
    ax2.set_xlim([0.4, 1])
    
    ax2.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label=r"$y_e$: Euler's method", color='red')
    
    ax2.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label=r"$y_{ei}$: Euler's improved method", color='blue')
    
    ax2.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label=r"$y_{rk}$: Runge Kutta 4th order", color='green')
    
    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)
    
    ax3 = plt.subplot(gs[1, :])
    ax3.set_title(r'$\Delta x = 0.05$')
    ax3.set_xlim([0.4, 1])
    
    ax3.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label=r"$y_e$: Euler's method", color='red')
    
    ax3.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label=r"$y_{ei}$: Euler's improved method", color='blue')
    
    ax3.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label=r"$y_{rk}$: Runge Kutta 4th order", color='green')
    
    plt.suptitle(r'Numerical solution to $\;\frac{dy}{dx} = y^2$', fontsize=12)
    plt.legend(bbox_to_anchor=(0.5, 2.4), loc='upper center', ncol=3)
    plt.show()


if __name__ == '__main__':
    main()

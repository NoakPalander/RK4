#!venv/bin/python

from matplotlib import pyplot as plt


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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.2, target=1.0, derv=lambda _, y: y * y)


    ax1.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label="Euler's method", color='red')

    ax1.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label="Euler's improved method", color='blue')

    ax1.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label="Runge Kutta 4th order", color='green')

    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.1, target=1.0, derv=lambda _, y: y * y)

    ax2.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label="Euler's method", color='red')

    ax2.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label="Euler's improved method", color='blue')

    ax2.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label="Runge Kutta 4th order", color='green')

    euler_data = euler_method(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)
    deluxe_data = euler_deluxe_method(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)
    rk4_data = runge_kutta(initial=(0.0, 1.0), delta_x=0.05, target=1.0, derv=lambda _, y: y * y)

    ax3.plot(list(map(lambda item: item[0], euler_data)), list(map(lambda item: item[1], euler_data)),
             label="Euler's method", color='red')

    ax3.plot(list(map(lambda item: item[0], deluxe_data)), list(map(lambda item: item[1], deluxe_data)),
             label="Euler's improved method", color='blue')

    ax3.plot(list(map(lambda item: item[0], rk4_data)), list(map(lambda item: item[1], rk4_data)),
             label="Runge Kutta 4th order", color='green')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    #plt.savefig('figure.png')


if __name__ == '__main__':
    main()

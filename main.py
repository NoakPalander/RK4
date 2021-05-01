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
	pass


if __name__ == '__main__':
	main()

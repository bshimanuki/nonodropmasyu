import sys

import numpy as np
import z3


BLOCKS = [
	['░░', '▒▒'],
	['▓▓', '██'],
]


class Nonogram:
	def __init__(self, s):
		self.cell_values = np.array([[not s for s in line.split('\t')] for line in s.split('\n')], dtype=bool)
		self.m, self.n = self.cell_values.shape
		self.rows = []
		for y in range(self.m):
			self.rows.append([])
			count = 0
			for x in range(self.n):
				if self.cell_values[y, x]:
					count += 1
				else:
					if count:
						self.rows[-1].append(count)
						count = 0
			if count:
				self.rows[-1].append(count)
		self.columns = []
		for x in range(self.n):
			self.columns.append([])
			count = 0
			for y in range(self.m):
				if self.cell_values[y, x]:
					count += 1
				else:
					if count:
						self.columns[-1].append(count)
						count = 0
			if count:
				self.columns[-1].append(count)
		self.solver = z3.Solver()
		self.rows_idx = []
		for y, row in enumerate(self.rows):
			self.rows_idx.append([])
			for j, v in enumerate(row):
				var = z3.Int(f'row_idx_{y}_{j}')
				self.rows_idx[-1].append(var)
				if j:
					self.solver.add(var >= self.rows_idx[-1][-2] + self.rows[y][j-1] + 1)
				else:
					self.solver.add(var >= 0)
				self.solver.add(var <= self.n - v)
		self.columns_idx = []
		for x, column in enumerate(self.columns):
			self.columns_idx.append([])
			for i, v in enumerate(column):
				var = z3.Int(f'column_idx_{x}_{i}')
				self.columns_idx[-1].append(var)
				if i:
					self.solver.add(var >= self.columns_idx[-1][-2] + self.columns[x][i-1] + 1)
				else:
					self.solver.add(var >= 0)
				self.solver.add(var <= self.m - v)
		self.cells = np.zeros_like(self.cell_values, dtype=np.object)
		for y, x in np.ndindex(self.cells.shape):
			var = z3.Bool(f'cell_{y}_{x}')
			self.cells[y, x] = var
			self.solver.add(var == z3.Or([
				z3.And(self.rows_idx[y][j] <= x, x < self.rows_idx[y][j] + self.rows[y][j])
				for j in range(len(self.rows[y]))
			]))
			self.solver.add(var == z3.Or([
				z3.And(self.columns_idx[x][i] <= y, y < self.columns_idx[x][i] + self.columns[x][i])
				for i in range(len(self.columns[x]))
			]))

		# check givens
		self.solver.push()
		self.solver.add(z3.And([
			var == self.cell_values[y, x].item() for (y, x), var in np.ndenumerate(self.cells)
		]))
		assert self.solver.check() == z3.sat
		self.solver.pop()

		self.solver.add(z3.Not(z3.And([
			var == self.cell_values[y, x].item() for (y, x), var in np.ndenumerate(self.cells)
		])))

	def get_next(self):
		check = self.solver.check()
		if check != z3.sat:
			return check, None
		model = self.solver.model()
		self.solver.add(z3.Not(z3.And([
			var == model.eval(var, model_completion=True)
			for _, var in np.ndenumerate(self.cells)
		])))
		solution = np.zeros_like(self.cell_values)
		for idx, var in np.ndenumerate(self.cells):
			solution[idx] = model.eval(var, model_completion=True)
		return check, solution


	def run(self):
		max_sols = 5
		sols = 0
		first = None
		self.solver.push()
		for i in range(max_sols):
			check, solution = self.get_next()
			if check != z3.sat:
				break
			sols += 1
			if first is None:
				first = solution
		self.solver.pop()
		print(f"{self.m}x{self.n} grid:")
		for y, line in enumerate(self.cell_values):
			print(''.join(BLOCKS[v.item()][0] for x, v in enumerate(line)))
		print(f"Found {sols}{'+' if sols == max_sols else ''} additional solution{'' if sols == 1 else 's'}.{' Ended with unknown.' if check == z3.unknown else ''}")
		if first is not None:
			for y, line in enumerate(first):
				print(''.join(BLOCKS[v.item()][(v != self.cell_values[y, x]).item()] for x, v in enumerate(line)))
		locked = np.zeros_like(self.cell_values)
		for idx, var in np.ndenumerate(self.cells):
			self.solver.push()
			self.solver.add(var != self.cell_values[idx].item())
			locked[idx] = self.solver.check() == z3.unsat
			self.solver.pop()
		if first is not None:
			print()
			for y, line in enumerate(self.cell_values):
				print(''.join(BLOCKS[v.item()][(v ^ ~locked[y, x]).item()] for x, v in enumerate(line)))


def main():
	nonogram = Nonogram(sys.stdin.read())
	nonogram.run()


if __name__ == '__main__':
	main()

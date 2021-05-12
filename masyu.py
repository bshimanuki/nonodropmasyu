'''
Determine if a masyu with ambiguous stones has a unique solution.
'''
from collections import defaultdict
import itertools
import sys

import numpy as np
import termcolor
import z3


NW = '┘'
NE = '└'
SW = '┐'
SE = '┌'
NS = '│'
WE = '─'
B = '⬤'
W = '◯'

PATH_NONE = 0
PATH_LINE = 1
PATH_BLACK = 2
PATH_WHITE = 3

def symbol2path(s):
	if s in (NW, NE, SW, SE, NS, WE):
		return PATH_LINE
	if s == B:
		return PATH_BLACK
	if s == W:
		return PATH_WHITE
	if not s:
		return PATH_NONE
	raise ValueError()


def index(a, y, x, default=False, lambd=None):
	if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
		if lambd is not None:
			return z3.And(a[y, x], lambd())
		return a[y, x]
	return default


Order = z3.DeclareSort('Order')


class Masyu:
	def __init__(self, s):
		'''
		s is a concatentation of two tsvs separated by multiple newlines. The first
		is a mapping of ambiguities and the second is the reference solution.
		Assumes no row in the grid is blank.
		'''
		lines = [line.split('\t') for line in s.split('\n')]
		split_i = list(map(any, lines)).index(False)
		letters = lines[:split_i]
		solution = list(filter(any, lines[split_i:]))
		self.letters = np.array(letters, dtype=np.object)
		self.solution = np.array(solution, dtype=np.object)
		if self.letters.ndim != 2 or self.solution.ndim != 2:
			raise ValueError(f"{'x'.join(self.letters.shape)} letter grid and {'x'.join(self.solution.shape)} masyu grid are not 2 dimensional")
		if self.letters.shape != self.solution.shape:
			raise ValueError(f"{'x'.join(self.letters.shape)} letter grid does not match {'x'.join(self.solution.shape)} masyu grid")
		self.m, self.n = self.solution.shape

		self.solver = z3.Solver()
		self.v = np.zeros((self.m-1, self.n), dtype=np.object)
		self.h = np.zeros((self.m, self.n-1), dtype=np.object)
		self.path = np.zeros_like(self.solution)
		self.order = np.zeros_like(self.solution)
		self.start = z3.Int('start') # for connectivity
		self.solver.add(0 <= self.start, self.start < self.solution.size)
		self.Parent = z3.PartialOrder(Order, 0)
		x, y = z3.Consts('x y', Order)
		for y, x in np.ndindex(self.v.shape):
			self.v[y, x] = z3.Bool(f'v_{y}_{x}')
		for y, x in np.ndindex(self.h.shape):
			self.h[y, x] = z3.Bool(f'h_{y}_{x}')
		for y, x in np.ndindex(self.solution.shape):
			self.path[y, x] = z3.Int(f'path_{y}_{x}')
			self.solver.add(0 <= self.path[y, x], self.path[y, x] < 4)
			self.order[y, x] = z3.Const(f'order_{y}_{x}', Order)
		self.solver.add(np.not_equal(self.order[:,:-1], self.order[:,1:], dtype=np.object).flatten().tolist())
		self.solver.add(np.not_equal(self.order[:-1,:], self.order[1:,:], dtype=np.object).flatten().tolist())
		for y, x in np.ndindex(self.solution.shape):
			edges = [
				index(self.v, y-1, x),
				index(self.h, y, x-1),
				index(self.v, y, x),
				index(self.h, y, x),
			]
			idx = y * self.solution.shape[1] + x
			# each cell is degree 0 or 2
			self.solver.add(z3.If(
				self.path[y, x] == PATH_NONE,
				z3.Not(z3.Or(edges)), # degree 0
				z3.PbEq([(e, 1) for e in edges], 2), # degree 2
			))
			# connectivity
			self.solver.add(z3.Or(
				self.path[y, x] == PATH_NONE,
				self.start == idx,
				index(self.v, y-1, x, lambd=lambda:self.Parent(self.order[y-1, x], self.order[y, x])),
				index(self.h, y, x-1, lambd=lambda:self.Parent(self.order[y, x-1], self.order[y, x])),
				index(self.v, y, x, lambd=lambda:self.Parent(self.order[y+1, x], self.order[y, x])),
				index(self.h, y, x, lambd=lambda:self.Parent(self.order[y, x+1], self.order[y, x])),
			))

		self.stones = np.full_like(self.solution, None)
		self.black = [] # list of column lists of row index variables
		self.black_solution = []
		self.white = []
		self.white_solution = []
		self.ambiguities = np.zeros_like(self.letters)
		for x in range(self.letters.shape[1]):
			self.black.append([])
			self.black_solution.append([])
			self.white.append([])
			self.white_solution.append([])
			letter2set = defaultdict(set)
			xstones = []
			for y in range(self.letters.shape[0]):
				letter = self.letters[y, x]
				letter2set[letter].add(y)
				self.ambiguities[y, x] = letter2set[letter]
			for y in range(self.solution.shape[0]):
				assert self.solution[y, x] in ('', B, W, NW, NE, SW, SE, NS, WE)
				if self.solution[y, x] in (B, W):
					self.stones[y, x] = z3.Int(f'stone_{y}_{x}')
					if self.solution[y, x] == B:
						self.black[-1].append(self.stones[y, x])
						self.black_solution[-1].append(y)
					else:
						self.white[-1].append(self.stones[y, x])
						self.white_solution[-1].append(y)
					xstones.append(self.stones[y, x])
					# constraint for black / white stone
					self.solver.add(z3.Or([
						z3.And(
							self.stones[y, x] == _y,
							z3.And(
								self.path[_y, x] == PATH_BLACK,
								z3.Or(
									z3.And(index(self.h, _y, x-1), index(self.h, _y, x-2)),
									z3.And(index(self.h, _y, x), index(self.h, _y, x+1)),
								),
								z3.Or(
									z3.And(index(self.v, _y-1, x), index(self.v, _y-2, x)),
									z3.And(index(self.v, _y, x), index(self.v, _y+1, x)),
								),
							) if self.solution[y, x] == B else z3.And(
								self.path[_y, x] == PATH_WHITE,
								z3.Or(
									z3.And(
										index(self.h, _y, x-1),
										index(self.h, _y, x),
										z3.Not(z3.And(index(self.h, _y, x-2), index(self.h, _y, x+1))),
									),
									z3.And(
										index(self.v, _y-1, x),
										index(self.v, _y, x),
										z3.Not(z3.And(index(self.v, _y-2, x), index(self.v, _y+1, x))),
									)
								),
							),
						)
						for _y in self.ambiguities[y, x]
					]))
			# ensure ambiguous stones don't overlap
			if xstones:
				self.solver.add(z3.Distinct(xstones))
		for (y, x), path in np.ndenumerate(self.path):
			self.solver.add(z3.Implies(path == PATH_BLACK, z3.Or([self.stones[_y, x] == y for _y in self.ambiguities[y, x] if self.solution[_y, x] == B])))
			self.solver.add(z3.Implies(path == PATH_WHITE, z3.Or([self.stones[_y, x] == y for _y in self.ambiguities[y, x] if self.solution[_y, x] == W])))

		# check givens
		self.v_solution = np.full_like(self.v, None)
		for y, x in np.ndindex(self.v_solution.shape):
			if any((
				self.solution[y, x] in (SW, SE, NS),
				index(self.solution, y+1, x) in (NW, NE, NS),
				self.solution[y, x] == W and index(self.solution, y-1, x) in (SW, SE, NS),
			)):
				self.v_solution[y, x] = True
			elif self.solution[y, x] not in (B, W) or self.solution[y+1, x] not in (B, W):
				self.v_solution[y, x] = False
		self.h_solution = np.full_like(self.h, None)
		for y, x in np.ndindex(self.h_solution.shape):
			if any((
				self.solution[y, x] in (NE, SE, WE),
				index(self.solution, y, x+1) in (NW, SW, WE),
				self.solution[y, x] == W and index(self.solution, y, x-1) in (NE, SE, WE),
			)):
				self.h_solution[y, x] = True
			elif self.solution[y, x] not in (B, W) or self.solution[y, x+1] not in (B, W):
				self.h_solution[y, x] = False
		self.path_solution = np.zeros_like(self.solution)
		for (y, x), symbol in np.ndenumerate(self.solution):
			self.path_solution[y, x] = symbol2path(symbol)
		_vars = []
		vals = []
		v, v_sol = zip(*((v, v_sol) for v, v_sol in zip(self.v.flat, self.v_solution.flat) if v_sol is not None))
		_vars.append(v)
		vals.append(v_sol)
		h, h_sol = zip(*((h, h_sol) for h, h_sol in zip(self.h.flat, self.h_solution.flat) if h_sol is not None))
		_vars.append(h)
		vals.append(h_sol)
		_vars.append(self.path.flat)
		vals.append(self.path_solution.flat)
		_vars.append(list(itertools.chain(*self.black, *self.white)))
		vals.append(list(itertools.chain(*self.black_solution, *self.white_solution)))
		constraints = z3.And([var == val for var, val in zip(itertools.chain(*_vars), itertools.chain(*vals))])
		self.solver.push()
		self.solver.add(constraints)
		check, solution = self.get_next()
		print(f"{self.m}x{self.n} grid:")
		assert check == z3.sat
		print(self.str(solution.v, solution.h, solution.path))
		check, _ = self.get_next()
		assert check == z3.unsat
		self.solver.pop()

		_vars = []
		vals = []
		_vars.append(self.v.flat)
		vals.append(solution.v.flat)
		_vars.append(self.h.flat)
		vals.append(solution.h.flat)
		_vars.append(self.path.flat)
		vals.append(solution.path.flat)
		constraints = z3.And([var == val for var, val in zip(itertools.chain(*_vars), itertools.chain(*vals))])
		self.solver.add(z3.Not(constraints))

	def get_next(self):
		check = self.solver.check()
		if check != z3.sat:
			return check, None
		model = self.solver.model()
		_vars = []
		_vars.append(self.v.flat)
		_vars.append(self.h.flat)
		_vars.append(self.path.flat)
		self.solver.add(z3.Not(z3.And([
			var == model.eval(var, model_completion=True)
			for var in itertools.chain(*_vars)
		])))
		solution = lambda: None
		solution.v = np.zeros_like(self.v)
		for idx, var in np.ndenumerate(self.v):
			solution.v[idx] = model.eval(var, model_completion=True)
		solution.h = np.zeros_like(self.h)
		for idx, var in np.ndenumerate(self.h):
			solution.h[idx] = model.eval(var, model_completion=True)
		solution.path = np.zeros_like(self.path)
		for idx, var in np.ndenumerate(self.path):
			solution.path[idx] = model.eval(var, model_completion=True)
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
		print(f"Found {sols}{'+' if sols == max_sols else ''} additional solution{'' if sols == 1 else 's'}.{' Ended with unknown.' if check == z3.unknown else ''}")
		if first is not None:
			print(self.str(first.v, first.h, first.path))

	@staticmethod
	def str(v, h, path):
		solution = np.full((v.shape[0]+1, v.shape[1]+1), ' ', dtype=np.object)
		for y, x in np.ndindex(solution.shape):
			if index(v, y-1, x) and index(h, y, x-1):
				solution[y, x] = NW
			if index(v, y-1, x) and index(h, y, x):
				solution[y, x] = NE
			if index(v, y, x) and index(h, y, x-1):
				solution[y, x] = SW
			if index(v, y, x) and index(h, y, x):
				solution[y, x] = SE
			if index(v, y-1, x) and index(v, y, x):
				solution[y, x] = NS
			if index(h, y, x-1) and index(h, y, x):
				solution[y, x] = WE
		for (y, x), p in np.ndenumerate(path):
			if p == PATH_BLACK:
				solution[y, x] = B
			if p == PATH_WHITE:
				solution[y, x] = W
		return '\n'.join(''.join(f'{c} ' if c in (B, W) else f' {c}' for c in line) for line in solution)

	def __str__(self):
		return self.str(self.v_solution, self.h_solution, self.path_solution)


def main():
	masyu = Masyu(sys.stdin.read())
	masyu.run()


if __name__ == '__main__':
	main()

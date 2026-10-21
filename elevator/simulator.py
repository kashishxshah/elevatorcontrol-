"""
Multi-elevator building simulator.

Time moves in discrete ticks. In one tick a car can move one floor.
Stopping at a floor (doors open, people get on and off) takes DWELL ticks.

How control is split
--------------------
* Each CAR runs the classical LOOK algorithm on its own list of stops:
  keep going in the current direction while there are stops ahead,
  otherwise turn around, and go idle when there is nothing to do.
* A DISPATCHER (the thing we compare, and the thing RL learns) decides
  which car answers each NEW hall call. A hall call is a lit button on a
  floor, e.g. "someone on floor 7 wants to go down".

So the decision problem is: "a new hall call has appeared, which car
should take it?" That is the action of our MDP.

Simplifications (also listed in the report's limitations):
* Only the assigned car picks up people from a hall call.
* Travel and door times are fixed, no acceleration.
* Passengers arrive as a Poisson process (see traffic.py).
"""

import numpy as np

UP, DOWN = 0, 1                 # index of a hall button direction
DIR_SIGN = {UP: +1, DOWN: -1}   # how that direction moves a car


class Passenger:
    __slots__ = ("origin", "dest", "dir", "t_arrive", "t_board", "t_done")

    def __init__(self, origin, dest, t):
        self.origin = origin
        self.dest = dest
        self.dir = UP if dest > origin else DOWN
        self.t_arrive = t
        self.t_board = None
        self.t_done = None


class Car:
    def __init__(self, idx, floor, capacity):
        self.idx = idx
        self.floor = floor
        self.direction = 0      # +1 up, -1 down, 0 idle
        self.capacity = capacity
        self.riders = []        # passengers inside the car
        self.car_calls = set()  # floors requested from inside the car
        self.assigned = set()   # hall calls (floor, dir) given to this car
        self.dwell = 0          # ticks left with doors open
        self.job_time = {}      # floor -> when the oldest request there was made

    def targets(self):
        """All floors this car still has to visit."""
        return self.car_calls | {f for (f, _) in self.assigned}

    def has_target_ahead(self, exclude_call=None):
        if self.direction == 0:
            return False
        for f in self.car_calls:
            if (f - self.floor) * self.direction > 0:
                return True
        for call in self.assigned:
            if call != exclude_call and (call[0] - self.floor) * self.direction > 0:
                return True
        return False


class Building:
    """
    Parameters
    ----------
    n_floors, n_cars : building size
    traffic          : a traffic.Traffic object that generates passengers
    car_policy       : how each car orders its own stops:
                       "look" (default), "fcfs" (oldest request first) or
                       "sstf" (shortest seek time first: nearest request first).
                       FCFS and SSTF are only used in the single-car baseline.
    fairness_lambda  : blends the two cost functions (see tick_cost)
                       0 -> total waiting time   (pure efficiency)
                       1 -> total SQUARED waiting time (punishes long waits)
    """

    DWELL = 2               # ticks the doors stay open at a stop
    OVERDUE = 40            # a wait longer than this counts as "overdue"
    SQUARE_SCALE = 20.0     # keeps the squared cost on a similar scale

    def __init__(self, n_floors, n_cars, traffic, capacity=8,
                 fairness_lambda=0.0, seed=0, car_policy="look"):
        assert car_policy in ("look", "fcfs", "sstf"), car_policy
        self.car_policy = car_policy
        self.F = n_floors
        self.N = n_cars
        self.traffic = traffic
        self.capacity = capacity
        self.lam = fairness_lambda
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        # spread the cars evenly over the building at the start
        starts = np.linspace(0, self.F - 1, self.N).round().astype(int)
        self.cars = [Car(i, int(s), self.capacity) for i, s in enumerate(starts)]
        self.waiting = [[[], []] for _ in range(self.F)]   # waiting[floor][dir]
        self.call_owner = {}                                # (floor, dir) -> car idx
        self.passengers = []                                # everyone ever created
        self.cost_log = []                                  # cost paid at each tick

    # ------------------------------------------------------------------
    def run(self, dispatcher, n_ticks):
        """Simulate n_ticks. dispatcher(building, call) must return a car index."""
        for _ in range(n_ticks):
            self.step(dispatcher)

    def step(self, dispatcher):
        new_calls = []

        # 1. new passengers arrive
        for p in self.traffic.sample(self.t, self.F, self.rng):
            self.passengers.append(p)
            self.waiting[p.origin][p.dir].append(p)
            call = (p.origin, p.dir)
            if call not in self.call_owner and call not in new_calls:
                new_calls.append(call)

        # 2. the dispatcher assigns every new hall call to a car
        for call in new_calls:
            self._assign(call, dispatcher)

        # 3. every car does one tick of LOOK
        for car in self.cars:
            self._move_car(car, dispatcher)

        # 4. pay the waiting cost for this tick
        self.cost_log.append(self.tick_cost())
        self.t += 1

    def _assign(self, call, dispatcher):
        k = int(dispatcher(self, call))
        self.call_owner[call] = k
        self.cars[k].assigned.add(call)
        self.cars[k].job_time.setdefault(call[0], self.t)

    # ------------------------------------------------------------------
    def _move_car(self, car, dispatcher):
        if car.dwell > 0:
            car.dwell -= 1
            return
        if self.car_policy != "look":
            return self._move_car_simple(car, dispatcher)

        f = car.floor
        stopped = False

        # people whose destination is this floor get off
        if f in car.car_calls:
            staying = []
            for p in car.riders:
                if p.dest == f:
                    p.t_done = self.t
                else:
                    staying.append(p)
            car.riders = staying
            car.car_calls.discard(f)
            stopped = True

        # pick up people from a hall call assigned to this car, if LOOK allows
        for d in (UP, DOWN):
            call = (f, d)
            if call not in car.assigned or len(car.riders) >= car.capacity:
                continue
            same_way = car.direction in (0, DIR_SIGN[d])
            if same_way or not car.has_target_ahead(exclude_call=call):
                self._board(car, call, dispatcher)
                stopped = True
                break           # serve one direction per stop

        if stopped:
            if f not in car.targets():
                car.job_time.pop(f, None)
            car.dwell = self.DWELL - 1
            return

        # otherwise move one floor following LOOK
        targets = car.targets()
        if not targets:
            car.direction = 0
            return
        if car.direction == 0 or not car.has_target_ahead():
            # turn towards the nearest remaining target
            nearest = min(targets, key=lambda x: abs(x - f))
            if nearest == f:
                return          # will be served next tick
            car.direction = 1 if nearest > f else -1
        car.floor += car.direction

    def _move_car_simple(self, car, dispatcher):
        """
        FCFS / SSTF for the single-car baseline. The car stops for anyone at
        its current floor (both directions), then heads to its next target:
        FCFS -> the floor with the oldest request, SSTF -> the nearest floor.
        """
        f = car.floor
        stopped = False
        if f in car.car_calls:
            staying = []
            for p in car.riders:
                if p.dest == f:
                    p.t_done = self.t
                else:
                    staying.append(p)
            car.riders = staying
            car.car_calls.discard(f)
            stopped = True
        for d in (UP, DOWN):
            if (f, d) in car.assigned and len(car.riders) < car.capacity:
                self._board(car, (f, d), dispatcher)
                stopped = True
        if f not in car.targets():
            car.job_time.pop(f, None)
        if stopped:
            car.dwell = self.DWELL - 1
            return

        targets = car.targets()
        if not targets:
            car.direction = 0
            return
        if self.car_policy == "fcfs":
            goal = min(targets, key=lambda x: (car.job_time.get(x, self.t), abs(x - f)))
        else:
            goal = min(targets, key=lambda x: (abs(x - f), car.job_time.get(x, self.t)))
        if goal == f:
            return
        car.direction = 1 if goal > f else -1
        car.floor += car.direction

    def _board(self, car, call, dispatcher):
        f, d = call
        queue = self.waiting[f][d]
        space = car.capacity - len(car.riders)
        boarding, left_behind = queue[:space], queue[space:]
        for p in boarding:
            p.t_board = self.t
            car.riders.append(p)
            car.car_calls.add(p.dest)
            car.job_time.setdefault(p.dest, self.t)
        car.assigned.discard(call)
        del self.call_owner[call]
        car.direction = DIR_SIGN[d]
        self.waiting[f][d] = left_behind
        if left_behind:                 # car was full: button is pressed again
            self._assign(call, dispatcher)

    # ------------------------------------------------------------------
    def waiting_times(self):
        """Current waits of everyone still standing at a hall button."""
        return [self.t - p.t_arrive
                for fl in self.waiting for q in fl for p in q]

    def tick_cost(self):
        """
        Cost paid this tick, summed over waiting passengers.

        Paying 1 per waiting passenger per tick adds up to their total wait W.
        Paying (2w + 1) at a wait of w adds up to W squared.
        The squared version makes one person waiting 60 ticks cost far
        more than six people waiting 10 ticks, which is our fairness lever.
        fairness_lambda blends the two.
        """
        waits = self.waiting_times()
        if not waits:
            return 0.0
        n = len(waits)
        linear = n
        squared = (2 * sum(waits) + n) / self.SQUARE_SCALE
        return (1 - self.lam) * linear + self.lam * squared

    def overdue(self):
        waits = self.waiting_times()
        return int(bool(waits) and max(waits) > self.OVERDUE)

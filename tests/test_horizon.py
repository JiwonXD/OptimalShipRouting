import math

import numpy as np
import shapely

from shiprouting.fuel.ships import KCS_SEAKEEPING, KCS_SHIP
from shiprouting.fuel.waves import AddedResistanceTable
from shiprouting.route.fuel import KNOT, FuelCost
from shiprouting.route.horizon import replan
from shiprouting.sea.land import Land
from shiprouting.sea.quadtree import Graph, astar
from test_fuel_route import T0, forecast

GRAPH = Graph.build(Land([shapely.box(100.0, 30.0, 101.0, 31.0)]), root=1.0, min_step=0.5, lat_limits=(-20.0, 20.0))
WAVES = AddedResistanceTable.build(KCS_SEAKEEPING, 16 * KNOT, T01=(7.0, 9.0), angle=(0.0, math.pi / 2, math.pi))


def first_plan():
    fc = forecast(Hs=2.0, wave_from_x=1.0, T_m10=8.0)
    cost = FuelCost(KCS_SHIP, fc, T0, 16 * KNOT, WAVES, frozen=False)
    s, g = GRAPH.nearest(0.5, 150.5), GRAPH.nearest(0.5, 170.5)  # about 2,200 km: some 75 h at 16 kn
    return g, astar(GRAPH, s, g, cost.costs(GRAPH), cost.link_cost, lambda n: 0.0)


def test_without_later_runs_the_plan_is_sailed_as_it_is():
    g, plan = first_plan()
    r = replan(GRAPH, KCS_SHIP, WAVES, 16 * KNOT, T0, g, plan, lambda t: None, 1, math.inf)
    assert r.lat.tolist() == plan.lat.tolist() and r.lon.tolist() == plan.lon.tolist()


def test_replanning_keeps_what_was_sailed_and_still_reaches_the_goal():
    g, plan = first_plan()
    asked = []

    def runs(t):
        asked.append(t)
        fc = forecast(Hs=2.0, wave_from_x=1.0, T_m10=8.0)
        return type(fc)(fc.lat, fc.lon, fc.time + (t - T0), fc.fields)

    r = replan(GRAPH, KCS_SHIP, WAVES, 16 * KNOT, T0, g, plan, runs, 1, math.inf)
    assert asked == [T0 + np.timedelta64(24, "h"), T0 + np.timedelta64(48, "h"), T0 + np.timedelta64(72, "h")]
    assert (r.lat[-1], r.lon[-1]) == (plan.lat[-1], plan.lon[-1])
    n = 1 + int(np.argmax(np.cumsum(np.hypot(np.diff(plan.lat), np.diff(plan.lon))) * 111.2 / (16 * 1.852) >= 24))
    assert r.lat[:n].tolist() == plan.lat[:n].tolist()  # the first day is sailed as first planned

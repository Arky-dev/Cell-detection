
import numpy as np
import ot
from scipy.spatial.distance import cdist

# Avec les cellules données comme ça :  [x, y, area]

def trajectoire_transport_optimal(cells_t, cells_next,
                max_distance=30,
                alpha=0.8,
                beta=0.2,
                reg=0.01,
                reg_m=1.0,
                min_fraction=0.15):

    pos_t = cells_t[:, :2]
    pos_next = cells_next[:, :2]
    area_t = cells_t[:, 2]
    area_next = cells_next[:, 2]

    n, m = len(cells_t), len(cells_next)
    if n == 0 or m == 0:
        return []

    a = area_t / area_t.sum()
    b = area_next / area_next.sum()

    distances = cdist(pos_t, pos_next)

    spatial_cost = distances / max_distance
    area_t = np.expand_dims(area_t, axis=1)
    area_next = np.expand_dims(area_next, axis=0)
    area_cost = np.abs(area_t - area_next) / (area_t + area_next)
    total_cost = alpha * spatial_cost + beta * area_cost

    plan = ot.unbalanced.sinkhorn_unbalanced(
        a, b, total_cost, reg, reg_m
    )
    plan[distances > max_distance] = 0

    
    row_sum = plan.sum(axis=1, keepdims=True)
    col_sum = plan.sum(axis=0, keepdims=True)
    row_fraction = plan / np.maximum(row_sum, 1e-12)
    col_fraction = plan / np.maximum(col_sum, 1e-12)

    matches = []
    for i in range(n):
        for j in range(m):
            if (row_fraction[i, j] >= min_fraction
                and col_fraction[i, j] >= min_fraction):

                matches.append((i, j))

    return matches, plan



def detect_fusion_division(matches, n, m):

    children = {i: [] for i in range(n)}
    parents = {j: [] for j in range(m)}

    for i, j in matches:
        children[i].append(j)
        parents[j].append(i)

    events = []

    for i, children_i in children.items():
        if len(children_i) > 1:
            events.append({
                "type": "division",
                "parent": i,
                "children": children_i
            })

    for j, parents_j in parents.items():
        if len(parents_j) > 1:
            events.append({
                "type": "fusion",
                "parents": parents_j,
                "child": j
            })

    return events


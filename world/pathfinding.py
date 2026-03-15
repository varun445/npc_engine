import heapq


def astar(start, goal, obstacles, rows, cols):
    """A* pathfinding on a 2-D grid.

    Args:
        start:     (row, col) of the starting cell.
        goal:      (row, col) of the destination cell.
        obstacles: set of (row, col) tuples that are impassable.
        rows:      total number of grid rows.
        cols:      total number of grid columns.

    Returns:
        A list of (row, col) tuples representing the path from *start*
        (exclusive) to *goal* (inclusive).  Returns an empty list when
        start == goal or when no path exists.
    """
    if start == goal:
        return []

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            r, c = neighbor
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if neighbor in obstacles:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return []  # No path found

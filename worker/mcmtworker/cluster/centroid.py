class CentroidState:
    TRACKED = 1
    LOST = 2
    DEAD = 3


class TargetCentroid:
    """Cluster centroid with state

    State transition machine:

    +-----------+ 5 cont miss   +------------+ exceed quota +--------------+
    |  Tracked  |<------------->|   Lost     |------------->|   Dead       |
    +-----------+    1 hit      +------------+              +--------------+
    """
    MISS_QUOTA = 60
    CONFIRM_THRESHOLD = 5

    def __init__(self, embedding):
        self.embedding = embedding
        self.state = CentroidState.TRACKED
        self._recent_actions = []
        self._miss_quota = TargetCentroid.MISS_QUOTA

    def __str__(self):
        if self.state == CentroidState.TRACKED: state = "tracked"
        if self.state == CentroidState.LOST: state = "lost"
        if self.state == CentroidState.DEAD: state = "dead"

        return "[{}]".format(state)

    def __repr__(self):
        return str(self)

    def hit(self):
        # Record recent actions
        self._miss_quota = TargetCentroid.MISS_QUOTA
        self._recent_actions.append("hit")
        if len(self._recent_actions) > TargetCentroid.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update state
        if self.state != CentroidState.DEAD:
            self.state = CentroidState.TRACKED

    def miss(self):
        # Record recent actions
        self._miss_quota -= 1
        self._recent_actions.append("miss")
        if len(self._recent_actions) > TargetCentroid.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update state
        if (
            self.state == CentroidState.TRACKED
            and len([ action
                    for action in self._recent_actions
                    if action == "miss" ]) >= TargetCentroid.CONFIRM_THRESHOLD
        ):
            self.state = CentroidState.LOST

        elif (
            self.state == CentroidState.LOST
            and self._miss_quota <= 0
        ):
            self.state = CentroidState.DEAD

    def is_dead(self):
        return self.state == CentroidState.DEAD

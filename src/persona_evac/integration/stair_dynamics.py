"""Existing capacity-based stair speed assumption; not an agent route oracle."""
def stair_speed_factor(stair,occupants):
    """Assumed density slowdown: 1.0 empty, 0.55 at model capacity."""
    ratio=min(1,max(0,occupants/max(1,stair['capacity'])))
    return 1-.45*ratio*ratio



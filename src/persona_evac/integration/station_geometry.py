"""Shared metric polygons for JuPedSim, FDS and the floor viewer."""
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def floor_geometry(floor, region=None):
    if "polygons" in floor:
        rings = floor["polygons"] if region is None else [floor["polygons"][region]]
        geom = unary_union([Polygon(ring) for ring in rings])
    else:
        rects = floor["walkable"] if region is None else [floor["walkable"][region]]
        geom = unary_union([box(x, y, x+w, y+h) for x, y, w, h in rects])
    for obstacle in floor.get("obstacles", []):
        geom = geom.difference(Polygon(obstacle["polygon"]) if "polygon" in obstacle
                               else box(*rect_bounds(obstacle["rect"])))
    return geom


def rect_bounds(rect):
    x, y, w, h = rect
    return x, y, x+w, y+h

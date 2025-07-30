from typing import Literal

from .interface.bound_triangles import BoundTrianglesFn
from .interface.composite import CompositeFn
from .interface.delineate_tiles import DelineateTilesFn
from .interface.generate_keys import GenerateKeysFn
from .interface.project_vertices import ProjectVerticesFn
from .interface.sort_keys import SortKeysFn
from .pytorch.bound_triangles import bound_triangles as bound_triangles_torch
from .pytorch.composite import composite as composite_torch
from .pytorch.delineate_tiles import delineate_tiles as delineate_tiles_torch
from .pytorch.generate_keys import generate_keys as generate_keys_torch
from .pytorch.project_vertices import project_vertices as project_vertices_torch
from .pytorch.sort_keys import sort_keys as sort_keys_torch
from .slang.bound_triangles import bound_triangles as bound_triangles_slang
from .slang.composite import composite as composite_slang
from .slang.delineate_tiles import delineate_tiles as delineate_tiles_slang
from .slang.generate_keys import generate_keys as generate_keys_slang
from .slang.project_vertices import project_vertices as project_vertices_slang
from .slang.sort_keys import sort_keys as sort_keys_slang

Backend = Literal["torch", "slang"]

PROJECT_VERTICES: dict[Backend, ProjectVerticesFn] = {
    "torch": project_vertices_torch,
    "slang": project_vertices_slang,
}

BOUND_TRIANGLES: dict[Backend, BoundTrianglesFn] = {
    "torch": bound_triangles_torch,
    "slang": bound_triangles_slang,
}

GENERATE_KEYS: dict[Backend, GenerateKeysFn] = {
    "torch": generate_keys_torch,
    "slang": generate_keys_slang,
}

SORT_KEYS: dict[Backend, SortKeysFn] = {
    "torch": sort_keys_torch,
    "slang": sort_keys_slang,
}

DELINEATE_TILES: dict[Backend, DelineateTilesFn] = {
    "torch": delineate_tiles_torch,
    "slang": delineate_tiles_slang,
}

COMPOSITE: dict[Backend, CompositeFn] = {
    "torch": composite_torch,
    "slang": composite_slang,
}

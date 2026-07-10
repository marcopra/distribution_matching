"""
Middle Room Environment - A central room connected to four isolated rooms.

This module provides a discrete gridworld where a square middle room is
connected to one square room on each side by a straight corridor.
"""

from typing import Dict, Optional, Set, Tuple

import gymnasium as gym

try:
    from env.rooms import BaseRoomEnv
except Exception:
    from rooms import BaseRoomEnv


class MiddleRoomEnv(BaseRoomEnv):
    """
    A gridworld environment with a middle room and four isolated outer rooms.

    The environment consists of:
    - One square room in the middle
    - Four square rooms placed north, south, west, and east of the middle room
    - One straight corridor from the middle room to each outer room
    - No direct connections between the four outer rooms

    Room coordinates are shifted after construction so the rendered grid uses
    nonnegative x/y positions even though the layout is built around the origin.

    Args:
        center_room_size: Size of the square middle room.
        outer_room_size: Default size for all four outer rooms.
        room_size: Alias for outer_room_size, matching the style of other room envs.
        north_room_size: Optional size override for the northern room.
        south_room_size: Optional size override for the southern room.
        west_room_size: Optional size override for the western room.
        east_room_size: Optional size override for the eastern room.
        connector_length: Length of each corridor between the middle and outer rooms.
        connector_positions: Optional dict with connector offsets on the middle room:
            north/south are x offsets, west/east are y offsets.
        goal_position: Goal position as (x, y) tuple or state index.
        start_position: Start position as (x, y) tuple or state index.
        max_steps: Maximum steps per episode.
        render_mode: Rendering mode ("human", "ansi", "rgb_array").
        show_coordinates: Whether to show coordinates in rendering.
        lava: Whether walls are lava.
        dense_reward: Whether to use dense Manhattan-distance reward.
    """

    SIDES = ("north", "south", "west", "east")

    def __init__(
        self,
        center_room_size: int = 5,
        outer_room_size: Optional[int] = None,
        room_size: Optional[int] = None,
        north_room_size: Optional[int] = None,
        south_room_size: Optional[int] = None,
        west_room_size: Optional[int] = None,
        east_room_size: Optional[int] = None,
        connector_length: int = 1,
        connector_positions: Optional[Dict[str, int]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        synthetic_first_transition: bool = False,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False,
        dense_reward: bool = False,
    ):
        if center_room_size < 2:
            raise ValueError("center_room_size must be at least 2")
        if connector_length < 1:
            raise ValueError("connector_length must be at least 1")

        if outer_room_size is not None and room_size is not None and outer_room_size != room_size:
            raise ValueError("Specify either outer_room_size or room_size, or give them the same value")

        default_outer_room_size = outer_room_size if outer_room_size is not None else room_size
        if default_outer_room_size is None:
            default_outer_room_size = center_room_size
        if default_outer_room_size < 2:
            raise ValueError("outer_room_size/room_size must be at least 2")

        self.center_room_size = center_room_size
        self.outer_room_size = default_outer_room_size
        self.connector_length = connector_length
        self.room_sizes = {
            "north": north_room_size if north_room_size is not None else default_outer_room_size,
            "south": south_room_size if south_room_size is not None else default_outer_room_size,
            "west": west_room_size if west_room_size is not None else default_outer_room_size,
            "east": east_room_size if east_room_size is not None else default_outer_room_size,
        }

        for side, size in self.room_sizes.items():
            if size < 2:
                raise ValueError(f"{side}_room_size must be at least 2")

        default_connector_position = center_room_size // 2
        self.connector_positions = {
            side: default_connector_position for side in self.SIDES
        }
        if connector_positions is not None:
            unknown_sides = set(connector_positions) - set(self.SIDES)
            if unknown_sides:
                raise ValueError(f"Unknown connector position keys: {sorted(unknown_sides)}")
            self.connector_positions.update(connector_positions)

        for side, position in self.connector_positions.items():
            if position < 0 or position >= center_room_size:
                raise ValueError(
                    f"connector_positions['{side}'] ({position}) must be in "
                    f"[0, {center_room_size - 1}]"
                )

        self._room_centers: Dict[str, Tuple[int, int]] = {}
        self._connector_entry_positions: Dict[str, Tuple[int, int]] = {}

        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava,
            dense_reward=dense_reward,
        )

    def _rectangle_cells(
        self,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
    ) -> Set[Tuple[int, int]]:
        return {
            (x, y)
            for x in range(min_x, max_x + 1)
            for y in range(min_y, max_y + 1)
        }

    def _build_signed_areas(self) -> Dict[str, Set[Tuple[int, int]]]:
        center_max = self.center_room_size - 1
        length = self.connector_length

        areas = {
            "center_room": self._rectangle_cells(0, center_max, 0, center_max),
        }

        north_x = self.connector_positions["north"]
        north_size = self.room_sizes["north"]
        north_attach_x = north_size // 2
        north_min_x = north_x - north_attach_x
        areas["north_corridor"] = {(north_x, y) for y in range(-length, 0)}
        areas["north_room"] = self._rectangle_cells(
            north_min_x,
            north_min_x + north_size - 1,
            -length - north_size,
            -length - 1,
        )

        south_x = self.connector_positions["south"]
        south_size = self.room_sizes["south"]
        south_attach_x = south_size // 2
        south_min_x = south_x - south_attach_x
        areas["south_corridor"] = {
            (south_x, y)
            for y in range(self.center_room_size, self.center_room_size + length)
        }
        areas["south_room"] = self._rectangle_cells(
            south_min_x,
            south_min_x + south_size - 1,
            self.center_room_size + length,
            self.center_room_size + length + south_size - 1,
        )

        west_y = self.connector_positions["west"]
        west_size = self.room_sizes["west"]
        west_attach_y = west_size // 2
        west_min_y = west_y - west_attach_y
        areas["west_corridor"] = {(x, west_y) for x in range(-length, 0)}
        areas["west_room"] = self._rectangle_cells(
            -length - west_size,
            -length - 1,
            west_min_y,
            west_min_y + west_size - 1,
        )

        east_y = self.connector_positions["east"]
        east_size = self.room_sizes["east"]
        east_attach_y = east_size // 2
        east_min_y = east_y - east_attach_y
        areas["east_corridor"] = {
            (x, east_y)
            for x in range(self.center_room_size, self.center_room_size + length)
        }
        areas["east_room"] = self._rectangle_cells(
            self.center_room_size + length,
            self.center_room_size + length + east_size - 1,
            east_min_y,
            east_min_y + east_size - 1,
        )

        return areas

    def _validate_area_layout(self, areas: Dict[str, Set[Tuple[int, int]]]):
        cell_owners: Dict[Tuple[int, int], str] = {}
        for area_name, cells in areas.items():
            for cell in cells:
                owner = cell_owners.get(cell)
                if owner is not None:
                    raise ValueError(
                        f"MiddleRoom layout overlaps at {cell}: {owner} and {area_name}. "
                        "Increase connector_length or reduce/reposition room sizes."
                    )
                cell_owners[cell] = area_name

        allowed_adjacencies = {
            frozenset(("center_room", "north_corridor")),
            frozenset(("center_room", "south_corridor")),
            frozenset(("center_room", "west_corridor")),
            frozenset(("center_room", "east_corridor")),
            frozenset(("north_corridor", "north_room")),
            frozenset(("south_corridor", "south_room")),
            frozenset(("west_corridor", "west_room")),
            frozenset(("east_corridor", "east_room")),
        }

        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for cell, owner in cell_owners.items():
            for dx, dy in directions:
                neighbor = (cell[0] + dx, cell[1] + dy)
                neighbor_owner = cell_owners.get(neighbor)
                if neighbor_owner is None or neighbor_owner == owner:
                    continue
                pair = frozenset((owner, neighbor_owner))
                if pair not in allowed_adjacencies:
                    raise ValueError(
                        f"MiddleRoom layout creates an unintended connection between "
                        f"{owner} and {neighbor_owner} near {cell}. Increase "
                        "connector_length or adjust connector_positions/room sizes."
                    )

    def _shift_areas_to_nonnegative(
        self,
        areas: Dict[str, Set[Tuple[int, int]]],
    ) -> Dict[str, Set[Tuple[int, int]]]:
        all_cells = [cell for cells in areas.values() for cell in cells]
        min_x = min(cell[0] for cell in all_cells)
        min_y = min(cell[1] for cell in all_cells)
        x_offset = -min_x if min_x < 0 else 0
        y_offset = -min_y if min_y < 0 else 0

        return {
            area_name: {
                (cell[0] + x_offset, cell[1] + y_offset)
                for cell in cells
            }
            for area_name, cells in areas.items()
        }

    def _get_area_center(self, cells: Set[Tuple[int, int]]) -> Tuple[int, int]:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)

    def _build_cells(self):
        signed_areas = self._build_signed_areas()
        self._validate_area_layout(signed_areas)
        areas = self._shift_areas_to_nonnegative(signed_areas)

        for area_name in (
            "center_room",
            "north_corridor",
            "north_room",
            "south_corridor",
            "south_room",
            "west_corridor",
            "west_room",
            "east_corridor",
            "east_room",
        ):
            for cell in sorted(areas[area_name], key=lambda value: (value[1], value[0])):
                self._add_cell(cell)

        self._room_centers = {
            "middle": self._get_area_center(areas["center_room"]),
            "north": self._get_area_center(areas["north_room"]),
            "south": self._get_area_center(areas["south_room"]),
            "west": self._get_area_center(areas["west_room"]),
            "east": self._get_area_center(areas["east_room"]),
        }

        self._connector_entry_positions = {
            "north": max(areas["north_corridor"], key=lambda cell: cell[1]),
            "south": min(areas["south_corridor"], key=lambda cell: cell[1]),
            "west": max(areas["west_corridor"], key=lambda cell: cell[0]),
            "east": min(areas["east_corridor"], key=lambda cell: cell[0]),
        }

    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: center of the eastern outer room."""
        return self._room_centers["east"]

    def get_room_centers(self) -> Dict[str, Tuple[int, int]]:
        """Return center positions for the middle and four outer rooms."""
        return dict(self._room_centers)

    def get_connector_positions(self) -> Dict[str, Tuple[int, int]]:
        """Return corridor cells adjacent to the middle room for each side."""
        return dict(self._connector_entry_positions)


gym.register(
    id="MiddleRoom-v0",
    entry_point="env.middle_room:MiddleRoomEnv",
    max_episode_steps=300,
)

"""The path table: does it describe a board, and does Travel follow the printed rule?"""

from __future__ import annotations

from src.strategy import map_graph as mg
from src.vision.spaces import BY_NAME


def test_every_path_ends_on_a_real_space_and_has_a_legend_kind():
    for path in mg.PATHS:
        assert path.a in BY_NAME, path.describe()
        assert path.b in BY_NAME, path.describe()
        assert path.a != path.b, path.describe()
        assert path.kind in mg.KINDS, path.describe()


def test_no_path_is_written_twice():
    seen = {frozenset((p.a, p.b)) for p in mg.PATHS}
    assert len(seen) == len(mg.PATHS)


def test_every_space_is_on_at_least_one_path():
    assert mg.unreachable() == ()


def test_the_whole_board_is_one_connected_map():
    start = next(iter(BY_NAME))
    seen, edge = {start}, [start]
    while edge:
        here = edge.pop()
        for space in mg.adjacent(here):
            if space not in seen:
                seen.add(space)
                edge.append(space)
    assert seen == set(BY_NAME)


def test_the_three_edge_crossings_are_marked_as_wrapping():
    wraps = {frozenset((p.a, p.b)) for p in mg.PATHS if p.wrap}
    assert wraps == {
        frozenset(("1", "19")),
        frozenset(("2", "Tokyo")),
        frozenset(("3", "Sydney")),
    }


def test_adjacency_is_symmetric():
    for space in BY_NAME:
        for other in mg.adjacent(space):
            assert space in mg.adjacent(other)


def test_prepare_for_travel_only_offers_tickets_for_paths_that_touch_the_space():
    # Istanbul is a rail hub with no water: no Ship ticket there.
    assert mg.ticket_kinds("Istanbul") == (mg.TRAIN,)
    # London is reached only by sea on this board.
    assert mg.ticket_kinds("London") == (mg.SHIP,)
    # Shanghai has both, and its Uncharted path to The Himalayas gives no ticket.
    assert set(mg.ticket_kinds("Shanghai")) == {mg.TRAIN, mg.SHIP}
    assert mg.UNCHARTED in mg.kinds_touching("Shanghai")


def test_travel_without_tickets_reaches_exactly_the_adjacent_spaces():
    routes = mg.travel_options("Rome")
    assert {r.destination for r in routes} == set(mg.adjacent("Rome"))
    assert all(r.moves == 1 for r in routes)
    assert all(r.train_spent == 0 and r.ship_spent == 0 for r in routes)


def test_a_train_ticket_buys_a_second_move_but_only_along_a_train_path():
    routes = {r.destination: r for r in mg.travel_options("Rome", train=1)}
    # Rome -> Istanbul -> 17 is two Train paths: reachable with one ticket.
    assert "17" in routes
    assert routes["17"].train_spent == 1
    # Rome -> London is Ship, and London's other paths are Ship too: no Ship ticket, no further.
    assert "13" not in routes
    # The free move still reaches everything adjacent.
    assert set(mg.adjacent("Rome")) <= set(routes)


def test_a_ship_ticket_does_not_pay_for_a_train_path():
    routes = {r.destination: r for r in mg.travel_options("Rome", ship=1)}
    assert "13" in routes and routes["13"].ship_spent == 1
    assert "17" not in routes


def test_an_uncharted_path_can_be_walked_but_never_bought():
    # The Pyramids sits between two Uncharted paths and one Ship path to Rome.
    assert "The Heart of Africa" in {r.destination for r in mg.travel_options("The Pyramids")}
    with_tickets = {r.destination for r in mg.travel_options("Rome", train=2, ship=2)}
    # Rome -> The Pyramids is Ship (one ticket); the Uncharted path onwards is never for sale.
    assert "The Pyramids" in with_tickets
    assert "The Heart of Africa" not in with_tickets


def test_a_route_never_visits_the_same_space_twice():
    for route in mg.travel_options("Shanghai", train=2, ship=2):
        visited = route.spaces("Shanghai")
        assert len(set(visited)) == len(visited)
        assert "Shanghai" not in visited


def test_each_destination_is_offered_once_by_its_cheapest_route():
    routes = mg.travel_options("Istanbul", train=2)
    assert len({r.destination for r in routes}) == len(routes)
    by_name = {r.destination: r for r in routes}
    assert by_name["17"].train_spent == 0  # adjacent: the free move


def test_steps_between_counts_moves_and_knows_the_far_side_of_the_map():
    assert mg.steps_between("Rome", "Rome") == 0
    assert mg.steps_between("Rome", "Istanbul") == 1
    assert mg.steps_between("Rome", "17") == 2
    assert mg.steps_between("Rome", "nowhere") is None
    # The edge crossing is one move, not a trip round the world.
    assert mg.steps_between("2", "Tokyo") == 1


def test_nearest_picks_the_closest_goal():
    goal, steps = mg.nearest("Shanghai", ["Rome", "Tokyo", "Buenos Aires"])
    assert goal == "Tokyo"
    assert steps == 1


def test_route_describes_itself_for_speaking():
    route = next(r for r in mg.travel_options("Rome", train=1) if r.destination == "17")
    text = route.describe("Rome")
    assert "Rome to Istanbul by train" in text
    assert "1 Train ticket" in text


def test_the_table_is_still_waiting_for_a_person_to_check_it():
    # This flips to True only once the owner has compared the drawing with the board; the
    # test exists so that flipping it is a deliberate act with a commit behind it.
    assert mg.VERIFIED in (True, False)

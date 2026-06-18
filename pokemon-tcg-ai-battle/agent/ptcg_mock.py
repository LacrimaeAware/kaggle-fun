"""A small, self-contained Pokemon-TCG-like environment.

This is NOT the real game and NOT the competition engine. It is a deliberately
simplified prize-race built to do three things before we have the organizer's engine:

  1. Exercise the always-legal contract: the policy must only ever pick a listed legal
     action and the game must terminate.
  2. Give the heuristic something with the right shape to be better than random
     (prizes, board presence, energy on attackers), so we can measure a first win rate.
  3. Pin the interface (Observation, legal action enumeration, step, clone) that the
     real Kaggle adapter will implement, so the policy code does not change when the real
     engine arrives.

Any win rate measured here has provenance `local-sim-MOCK`. It says the architecture
works and the heuristic dominates random in this toy. It says nothing about the real
game. Do not promote a mock number to a belief about the real contest.

Simplifications vs real PTCG: one prize per knockout, 3 prizes to win, bench up to 3,
one energy attach per turn, basics only (no evolution), fixed-damage attacks, no Trainer
cards, no weakness/resistance, no special conditions. The turn is a real sequence of
decisions (play, attach, retreat, then attack or pass), which is the part that matters
for testing a decision policy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Literal

# --- cards -------------------------------------------------------------------

@dataclass(frozen=True)
class Attack:
    name: str
    cost: int          # energy required attached to the active Pokemon
    damage: int


@dataclass(frozen=True)
class PokemonCard:
    name: str
    max_hp: int
    attacks: tuple[Attack, ...]


@dataclass(frozen=True)
class EnergyCard:
    name: str = "Energy"


Card = PokemonCard | EnergyCard


# A tiny card pool. The numbers are picked so that attacking is good but managing energy
# and bench matters (a one-energy poke that hits for a little, a two-energy poke that
# hits hard but needs setup).
POOL: tuple[PokemonCard, ...] = (
    PokemonCard("Sparker", 60, (Attack("Jolt", 1, 20),)),
    PokemonCard("Bruiser", 90, (Attack("Smash", 2, 50), Attack("Tap", 1, 10))),
    PokemonCard("Tank", 120, (Attack("Crush", 3, 70), Attack("Bump", 1, 10))),
)


# --- in-play state -----------------------------------------------------------

@dataclass
class InPlay:
    card: PokemonCard
    damage: int = 0           # damage counters on it
    energy: int = 0           # energy attached

    @property
    def hp_left(self) -> int:
        return self.card.max_hp - self.damage

    def clone(self) -> "InPlay":
        return InPlay(self.card, self.damage, self.energy)


@dataclass
class PlayerState:
    deck: list[Card]
    hand: list[Card]
    active: InPlay | None
    bench: list[InPlay]
    prizes_remaining: int          # prizes this player still needs to take to win
    energy_attached_this_turn: bool = False
    retreated_this_turn: bool = False
    lost: bool = False             # set on deck-out or no-Pokemon

    def clone(self) -> "PlayerState":
        return PlayerState(
            deck=list(self.deck), hand=list(self.hand),
            active=self.active.clone() if self.active else None,
            bench=[b.clone() for b in self.bench],
            prizes_remaining=self.prizes_remaining,
            energy_attached_this_turn=self.energy_attached_this_turn,
            retreated_this_turn=self.retreated_this_turn, lost=self.lost,
        )


BENCH_MAX = 3
START_PRIZES = 3
START_HAND = 5
DECK_SIZE = 24


# --- actions -----------------------------------------------------------------
# An action is a small immutable tuple. The policy never constructs one; it only ever
# returns an element of legal_actions(state).

Action = tuple
# ("play_basic", hand_index)
# ("attach", target)          target = -1 active, or bench index 0..n
# ("retreat", bench_index)
# ("attack", attack_index)    ends the turn
# ("pass",)                   ends the turn


@dataclass
class GameState:
    players: list[PlayerState]
    to_move: int               # 0 or 1
    turn: int = 0
    rng_seed: int = 0
    winner: int | None = None

    def clone(self) -> "GameState":
        return GameState(
            players=[p.clone() for p in self.players],
            to_move=self.to_move, turn=self.turn, rng_seed=self.rng_seed,
            winner=self.winner,
        )


# --- observation (what a player is allowed to see) ---------------------------

@dataclass(frozen=True)
class Observation:
    """The public state plus the moving player's own hand. Hidden: opponent hand and
    deck order, which is what makes the real game imperfect-information. The mock exposes
    full state to the simulator-based policy via the env; this Observation is the shape
    the real adapter must fill, and is what a no-simulator policy would consume."""
    me: int
    turn: int
    my_hand: tuple[str, ...]
    my_active: tuple[str, int, int] | None      # (name, hp_left, energy)
    my_bench: tuple[tuple[str, int, int], ...]
    opp_active: tuple[str, int, int] | None
    opp_bench: tuple[tuple[str, int, int], ...]
    my_prizes_remaining: int
    opp_prizes_remaining: int
    my_deck_size: int
    opp_hand_size: int


def observe(state: GameState, player: int) -> Observation:
    me, opp = state.players[player], state.players[1 - player]

    def pub(ip: InPlay | None):
        return (ip.card.name, ip.hp_left, ip.energy) if ip else None

    return Observation(
        me=player, turn=state.turn,
        my_hand=tuple(c.name for c in me.hand),
        my_active=pub(me.active),
        my_bench=tuple(pub(b) for b in me.bench),  # type: ignore[arg-type]
        opp_active=pub(opp.active),
        opp_bench=tuple(pub(b) for b in opp.bench),  # type: ignore[arg-type]
        my_prizes_remaining=me.prizes_remaining,
        opp_prizes_remaining=opp.prizes_remaining,
        my_deck_size=len(me.deck),
        opp_hand_size=len(opp.hand),
    )


# --- environment -------------------------------------------------------------

class MockPTCG:
    """Engine for the toy game. Pure functions over GameState where practical; chance
    (draws) is resolved with a seeded RNG carried on the state for reproducibility."""

    def new_game(self, seed: int = 0) -> GameState:
        rng = random.Random(seed)
        players = [self._new_player(rng) for _ in range(2)]
        state = GameState(players=players, to_move=0, turn=1, rng_seed=seed)
        # both players start with an active basic already promoted
        for p in state.players:
            self._ensure_active(p, rng)
        return state

    def _new_player(self, rng: random.Random) -> PlayerState:
        deck: list[Card] = []
        for _ in range(DECK_SIZE):
            deck.append(rng.choice(POOL) if rng.random() < 0.45 else EnergyCard())
        rng.shuffle(deck)
        hand = [deck.pop() for _ in range(START_HAND)]
        # guarantee at least one basic in the opening hand (mock mulligan)
        if not any(isinstance(c, PokemonCard) for c in hand):
            for i, c in enumerate(deck):
                if isinstance(c, PokemonCard):
                    hand[0], deck[i] = c, hand[0]
                    break
        return PlayerState(deck=deck, hand=hand, active=None, bench=[],
                           prizes_remaining=START_PRIZES)

    def _ensure_active(self, p: PlayerState, rng: random.Random) -> None:
        if p.active is None:
            for i, c in enumerate(p.hand):
                if isinstance(c, PokemonCard):
                    p.active = InPlay(c)
                    p.hand.pop(i)
                    return

    # -- turn lifecycle --
    def _draw(self, p: PlayerState) -> None:
        if not p.deck:
            p.lost = True       # deck-out
            return
        p.hand.append(p.deck.pop())

    def start_turn(self, state: GameState) -> None:
        p = state.players[state.to_move]
        p.energy_attached_this_turn = False
        p.retreated_this_turn = False
        self._draw(p)
        if p.lost:
            state.winner = 1 - state.to_move

    # -- legality --
    def legal_actions(self, state: GameState) -> list[Action]:
        if state.winner is not None:
            return []
        p = state.players[state.to_move]
        actions: list[Action] = []
        if p.active is None:
            # must promote: only play_basic actions
            for i, c in enumerate(p.hand):
                if isinstance(c, PokemonCard):
                    actions.append(("play_basic", i))
            return actions or [("pass",)]
        # play a basic to the bench
        if len(p.bench) < BENCH_MAX:
            for i, c in enumerate(p.hand):
                if isinstance(c, PokemonCard):
                    actions.append(("play_basic", i))
        # attach one energy per turn
        if not p.energy_attached_this_turn and any(isinstance(c, EnergyCard) for c in p.hand):
            actions.append(("attach", -1))
            for b in range(len(p.bench)):
                actions.append(("attach", b))
        # retreat (swap active with a bench Pokemon), once per turn
        if not p.retreated_this_turn:
            for b in range(len(p.bench)):
                actions.append(("retreat", b))
        # attack (ends the turn) if any attack is affordable
        for ai, atk in enumerate(p.active.card.attacks):
            if p.active.energy >= atk.cost:
                actions.append(("attack", ai))
        actions.append(("pass",))
        return actions

    # -- transition --
    def step(self, state: GameState, action: Action) -> GameState:
        """Apply an action, returning a NEW state (clone-and-mutate). Attacks and pass
        end the turn (draw for the next player happens here)."""
        s = state.clone()
        p = s.players[s.to_move]
        kind = action[0]

        if kind == "play_basic":
            i = action[1]
            card = p.hand.pop(i)
            assert isinstance(card, PokemonCard)
            if p.active is None:
                p.active = InPlay(card)
            else:
                p.bench.append(InPlay(card))
            return s

        if kind == "attach":
            target = action[1]
            for i, c in enumerate(p.hand):
                if isinstance(c, EnergyCard):
                    p.hand.pop(i)
                    break
            tgt = p.active if target == -1 else p.bench[target]
            if tgt:
                tgt.energy += 1
            p.energy_attached_this_turn = True
            return s

        if kind == "retreat":
            b = action[1]
            p.active, p.bench[b] = p.bench[b], p.active  # type: ignore[assignment]
            p.retreated_this_turn = True
            return s

        if kind == "attack":
            self._resolve_attack(s, action[1])
            if s.winner is None:
                self._end_turn(s)
            return s

        if kind == "pass":
            self._end_turn(s)
            return s

        raise ValueError(f"unknown action {action!r}")

    def _resolve_attack(self, s: GameState, attack_index: int) -> None:
        atk = s.players[s.to_move].active.card.attacks[attack_index]  # type: ignore[union-attr]
        defender = s.players[1 - s.to_move]
        if defender.active is None:
            return
        defender.active.damage += atk.damage
        if defender.active.hp_left <= 0:
            # knockout: the ATTACKER takes a prize
            attacker = s.players[s.to_move]
            attacker.prizes_remaining -= 1
            if attacker.prizes_remaining <= 0:
                s.winner = s.to_move
                return
            # defender must promote from bench, else they have no Pokemon and lose
            if defender.bench:
                defender.active = defender.bench.pop(0)
            else:
                defender.active = None
                defender.lost = True
                s.winner = s.to_move

    def _end_turn(self, s: GameState) -> None:
        s.to_move = 1 - s.to_move
        s.turn += 1
        self.start_turn(s)

    def is_terminal(self, state: GameState) -> bool:
        return state.winner is not None

    def winner(self, state: GameState) -> int | None:
        return state.winner

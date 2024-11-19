import random
import re
from typing import Dict, List, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

@register_env
class FindHuman(Environment):
    type_name = "findHuman"

    def __init__(
        self,
        human_player,
        player_names: List[str],
        **kwargs,
    ):
        super().__init__(player_names=player_names, **kwargs)

        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()

        self.human_name = human_player

        # Game states
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "ask questions"  # "give clues", "accuse", "guess"
        self._players_votes = None
        self._initialized = False

        self.reset()

    def get_next_player(self) -> str:
        """Get the next player."""
        return self.player_names[self._next_player_idx]

    def reset(self):
        """Reset the game"""
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "ask questions"

        self.message_pool.reset()

        self._moderator_speak(
            "Now everyone can ask eachother questions to determine who is the human. Be aware that the human can lie."
            f"You cannot repeat what others has said. We will start with {self.player_names[0]}."
        )
        self._current_turn = 1

        self._players_votes = {name: 0 for name in self.player_names}

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """Get observation for the player."""
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self._current_turn
            )

    def _text2vote(self, text) -> str:
        """Convert text to vote, return a player's name."""
        # lower = text.lower().replace("[", "").replace("]", "").replace(".", "")
        text = text.lower()
        for name in self.player_names:
            candidates = [
                name.lower(),
                name.lower().replace(" ", ""),
                name.lower().replace(" ", "_"),
            ]
            if any([candidate in text for candidate in candidates]):
                return name
        return ""

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """Moderator say something."""
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def get_rewards(self, human_win: bool) -> Dict[str, float]:
        """Get rewards for each player."""
        rewards = {}
        for name in self.player_names:
            # The winner gets 1, the loser gets 0
            rewards[name] = float((name == self.human_name) == human_win)

        return rewards

    def is_terminal(self) -> bool:
        """Check if the conversation is over."""
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content.startswith(
            SIGNAL_END_OF_CONVERSATION
        ):
            return True

    def step(self, player_name: str, action: str, max_steps) -> TimeStep:
        """
        Step function that is called by the arena.

        Args:
            player_name: the name of the player that takes the action
            action: the action that the agents wants to take
        """
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        # self.message_pool.print()
        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."
        if self._current_phase == "ask questions":
            message = Message(
                agent_name=player_name, content=action, turn=self._current_turn
            )
            self.message_pool.append_message(message)

            # Update the counters
            self._current_turn += 1
            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_phase = "accuse" if self._current_turn == max_steps-1 else "ask questions"
                self._moderator_speak(
                    "Now vote which of the other players (excluding yourself) is the human. "
                    "You cannot vote for yourself."
                )
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages
        elif self._current_phase == "accuse":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name],
            )
            self.message_pool.append_message(message)
            vote = self._text2vote(action)
            if vote in self.player_names:
                self._players_votes[vote] += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
                rewards = self.get_zero_rewards()
                terminal = False
            else:
                # print(self._players_votes)
                accuse_correct, even_vote = True, False
                max_vote_player = max(self._players_votes, key=self._players_votes.get)
                # detach if other players has the same number of votes
                for name, vote in self._players_votes.items():
                    if (
                        name != max_vote_player
                        and vote == self._players_votes[max_vote_player]
                    ):
                        accuse_correct, even_vote = False, True
                if max_vote_player != self.human_name:
                    accuse_correct = False

                if not accuse_correct:
                    if even_vote:
                        self._moderator_speak(
                            f"There are even votes. The accusation does not stand. "
                            f"{self.human_name} is the human. {self.human_name} won the game!"
                        )
                    else:
                        self._moderator_speak(
                            f"The most-voted player is {max_vote_player}. The accusation is incorrect. "
                            f"{self.human_name} is the human. {self.human_name} won the game!"
                        )
                    rewards = self.get_rewards(human_win=True)
                    terminal = True
                else:
                    self._moderator_speak(
                        f"The accusation is correct! {self.human_name} is the human! "
                    )
                    rewards = self.get_zero_rewards()
                    terminal = True
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(), reward=rewards, terminal=terminal
            )
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep

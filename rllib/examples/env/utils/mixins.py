##########
# Contribution by the Center on Long-Term Risk:
# https://github.com/longtermrisk/marltoolbox
##########
from abc import ABC
import numpy as np

from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface


class TwoPlayersTwoActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in a two player discrete game.
    Logs the frequency of each state.
    """

    def _init_info(self):
        self.cc_count = []
        self.dd_count = []
        self.cd_count = []
        self.dc_count = []

    def _reset_info(self):
        self.cc_count.clear()
        self.dd_count.clear()
        self.cd_count.clear()
        self.dc_count.clear()

    def _get_episode_info(self):
        return {
            "CC": np.mean(self.cc_count).item(),
            "DD": np.mean(self.dd_count).item(),
            "CD": np.mean(self.cd_count).item(),
            "DC": np.mean(self.dc_count).item(),
        }

    def _accumulate_info(self, ac0, ac1):
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)


class MessagedTwoPlayersTwoActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in a two player discrete game.
    Logs the frequency of each state.
    """

    def _init_info(self):
        self.cc_count = []
        self.dd_count = []
        self.cd_count = []
        self.dc_count = []
        self.turth_telling_count = []
        self.promise_breaking_count = []

    def _reset_info(self):
        self.cc_count.clear()
        self.dd_count.clear()
        self.cd_count.clear()
        self.dc_count.clear()
        self.turth_telling_count.clear()
        self.promise_breaking_count.clear()

    def _get_episode_info(self):
        return {
            "CC": np.mean(self.cc_count).item(),
            "DD": np.mean(self.dd_count).item(),
            "CD": np.mean(self.cd_count).item(),
            "DC": np.mean(self.dc_count).item(),
            # "Truth": np.mean(self.turth_telling_count).item(),
            # "PromiseBreak": np.mean(self.promise_breaking_count).item(),
        }

    def _accumulate_info(self, ac0, ac1, NUM_ACTIONS=2):
        actual_action_player_0 = ac0 // NUM_ACTIONS
        actual_action_player_1 = ac1 // NUM_ACTIONS
        message_action_player_0 = ac0 % NUM_ACTIONS
        message_action_player_1 = ac1 % NUM_ACTIONS
        self.cc_count.append(
            actual_action_player_0 == 0 and actual_action_player_1 == 0
        )
        self.cd_count.append(
            actual_action_player_0 == 0 and actual_action_player_1 == 1
        )
        self.dc_count.append(
            actual_action_player_0 == 1 and actual_action_player_1 == 0
        )
        self.dd_count.append(
            actual_action_player_0 == 1 and actual_action_player_1 == 1
        )


class NPlayersNDiscreteActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with
    discrete actions.
    Logs the frequency of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def _init_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _reset_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _get_episode_info(self):
        info = {}
        if self.info_counters["n_steps_accumulated"] > 0:
            for k, v in self.info_counters.items():
                if k != "n_steps_accumulated":
                    info[k] = v / self.info_counters["n_steps_accumulated"]

        return info

    def _accumulate_info(self, *actions):
        id = "_".join([str(a) for a in actions])
        if id not in self.info_counters:
            self.info_counters[id] = 0
        self.info_counters[id] += 1
        self.info_counters["n_steps_accumulated"] += 1

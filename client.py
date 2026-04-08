# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for InvoiceReconcileEnv."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import InvoicereconcileenvAction, InvoicereconcileenvObservation


class InvoicereconcileenvEnv(
    EnvClient[InvoicereconcileenvAction, InvoicereconcileenvObservation, State]
):
    def _step_payload(self, action: InvoicereconcileenvAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[InvoicereconcileenvObservation]:
        obs_data = payload.get("observation", {})
        observation = InvoicereconcileenvObservation(
            **obs_data,
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State.model_validate(payload)

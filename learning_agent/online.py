import uuid
from typing import Generic, TypeVar, Dict, Tuple

from converter import BaseConverter
from models import BaseModel

S = TypeVar("State")
A = TypeVar("Action")
IS = TypeVar("InternalState")
IA = TypeVar("InternalAction")

class OnlineAgent(Generic[S, A, IS, IA]):
    """Agent that wraps a model following the BaseModel interface."""

    def __init__(self, converter: BaseConverter[S, A, IS, IA], model: BaseModel[S, A]):
        self.converter = converter
        self.model = model
        self.pending: Dict[str, Tuple[IS, IA]] = {}

    def predict(self, state: S) -> Tuple[str, A]:
        state = self.converter.encode_state(state)
        action = self.model.predict_action(state)

        pid = str(uuid.uuid4())
        self.pending[pid] = (state, action)

        return pid, self.converter.decode_action(action)

    def learn(self, pid: str, reward: float) -> None:
        state, action = self.pending.pop(pid)
        self.model.update_model(state, action, reward)

    def reset(self) -> None:
        self.model.reset_model()
        self.pending.clear()

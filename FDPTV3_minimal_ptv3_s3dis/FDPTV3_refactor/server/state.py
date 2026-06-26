"""服务端运行时状态。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResumeState:
    round_idx: int = 0
    user_idx: int = 0


@dataclass
class ServerRuntimeState:
    total_rounds: int
    num_users: int
    resume: ResumeState
    current_round: int = 0
    successful_clients: int = 0
    global_model_path: Optional[str] = None

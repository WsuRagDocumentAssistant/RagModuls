#================================================
# __init__.py
#================================================
"""
LLM 프롬프트를 모아둔다.

기능마다 지시(system)와 데이터(user) 두 벌을 등록해 두고 get_prompt(키, 데이터) 로
한 쌍을 꺼낸다.
"""

from .prompt import SYSTEMS, USERS, get_prompt

__all__ = [
    "get_prompt",
    "SYSTEMS",
    "USERS",
]

from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from main import Window


default_marker_name = 'jf5Tg'
debug: bool = False
window: Union[Window, None] = None

import asyncio
import base64
import os
import shutil
from pathlib import Path
from typing import Literal, TypedDict, Optional, Dict, Any
from tools.base import NebiusTool

class NebiusComputerTool(NebiusTool):
    """Адаптированная версия ComputerTool для Nebius API"""

    def __init__(self):
        super().__init__()
        self._session = None
        self.name = "computer_control"
        self.description = "Control computer mouse, keyboard and take screenshots"
        self._screenshot_delay = 2.0
        self.width = int(os.getenv("WIDTH", "1920"))
        self.height = int(os.getenv("HEIGHT", "1080"))
        self.display_num = os.getenv("DISPLAY_NUM")
        self.display = os.getenv("DISPLAY", ":0")

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["mouse_move", "click", "type", "screenshot"],
                        "description": "Action to perform"
                    },
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "text": {"type": "string", "description": "Text to type"},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button"
                    }
                },
            "required": ["action"]
        }

    async def execute(self, **kwargs) -> Dict[str, str]:

        action = kwargs.get("action")
        # Standard format
        response = {
            "status": "success",  # or "error" if we have problems
            "command": action,  # original command for logs
            "output": "",  # stdout
            "error": "",  # stderr
            "exit_code": 0,
            "message": ""  # additional message (for errors)
        }

        try:
            if action == "screenshot":
                return await self._take_screenshot()
            elif action == "mouse_move":
                return await self._move_mouse(kwargs["x"], kwargs["y"])
            elif action == "click":
                return await self._click(kwargs["button"], kwargs.get("x"), kwargs.get("y"))
            elif action == "type":
                return await self._type_text(kwargs["text"])
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    async def _take_screenshot(self) -> Dict[str, str]:
        """Упрощенная версия screenshot для Nebius"""
        path = Path("/tmp/screenshot.png")
        cmd = f"DISPLAY=:{self.display_num} scrot -p {path}" if self.display_num else f"scrot -p {path}"

        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()

        if path.exists():
            with open(path, "rb") as f:
                return {"image": base64.b64encode(f.read()).decode()}
        return {"error": "Failed to take screenshot"}

    async def _move_mouse(self, x: int, y: int) -> Dict[str, str]:
        cmd = f"xdotool mousemove {x} {y}"
        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()
        return {"status": "success"}


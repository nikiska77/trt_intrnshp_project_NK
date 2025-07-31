import asyncio
import os
from typing import Any, Dict, Optional

# from .base import ToolError, ToolResult

class NebiusBashTool:
    """
    Bash Tool for Nebius AI API.
    1. Tools description is modified for Nebius (to_params)
    2. Inout and output data structure
    3. Errors handling according to Nebius style
    """

    def __init__(self):
        self._session = None
        self.command = "/bin/bash"
        self._timeout = 120.0
        self._sentinel = "<<exit>>"

    def to_params(self) -> Dict[str, Any]:
        return {
            "name": "execute_bash",
            "description": "Executes bash-commands in isolated environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command for execution"
                    },
                    "restart_session": {
                        "type": "boolean",
                        "default": False,
                        "description": "Forced restart of bash session"
                    }
                },
                "required": ["command"]
            }
        }

    async def __call__(self, command: str, restart_session: bool = False) -> Dict[str, str]:
        # I added dangerous commands handling
        blacklist = ["rm -rf", "dd", "shutdown"]
        if any(cmd in command for cmd in blacklist):
            return {"status": "error", "message": "Dangerous command blocked"}
        try:
            if restart_session or self._session is None:
                await self._restart_session()

            result = await self._execute_command(command)
            return {
                "status": "success",
                "output": result["output"],
                "error": result["error"]
            }
        # ToolResult(result)

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
        # ToolError(error=str(e)))

    async def _restart_session(self):
        if self._session:
            self._session.stop()

        self._session = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _execute_command(self, command: str) -> Dict[str, str]:
        """Command execution with timeout"""
        try:
            # Command send
            self._session.stdin.write(f"{command}; echo '{self._sentinel}'\n".encode())
            await self._session.stdin.drain()

            # Output read with timeout
            async with asyncio.timeout(self._timeout):
                output = await self._read_until_sentinel(self._session.stdout)
                error = await self._read_until_sentinel(self._session.stderr)

            return {
                "output": output.replace(self._sentinel, "").strip(),
                "error": error.replace(self._sentinel, "").strip()
            }

        except asyncio.TimeoutError:
            raise RuntimeError(f"Command was not executed within  {self._timeout} seconds")

    async def _read_until_sentinel(self, stream: asyncio.StreamReader) -> str:
        """Read until sentinel"""
        buffer = []
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if self._sentinel in decoded:
                buffer.append(decoded.replace(self._sentinel, ""))
                break
            buffer.append(decoded)
        return "\n".join(buffer)
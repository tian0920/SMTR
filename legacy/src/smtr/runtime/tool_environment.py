"""Realistic tool environment with tool registry.

Implements the EnvironmentAdapter protocol with richer actions
beyond the simple 3-step valid_sequence of ToyEnvironment.
"""

from copy import deepcopy
from typing import Any


class ToolEnvironment:
    """Realistic tool environment with tool registry."""

    TOOLS: dict[str, dict[str, Any]] = {
        "read_file": {
            "params": ["path"],
            "description": "Read content from a file",
        },
        "write_file": {
            "params": ["path", "content"],
            "description": "Write content to a file",
        },
        "search_web": {
            "params": ["query"],
            "description": "Search the web for information",
        },
        "run_command": {
            "params": ["command"],
            "description": "Execute a shell command",
        },
        "send_message": {
            "params": ["recipient", "message"],
            "description": "Send a message to a recipient",
        },
        "list_files": {
            "params": ["directory"],
            "description": "List files in a directory",
        },
        "delete_file": {
            "params": ["path"],
            "description": "Delete a file",
        },
        "create_directory": {
            "params": ["path"],
            "description": "Create a new directory",
        },
    }

    def __init__(self, *, seed: int = 7) -> None:
        self.seed = seed
        self._init_state()

    def _init_state(self) -> None:
        """Initialize environment state."""
        self._state: dict[str, Any] = {
            "filesystem": {
                "/workspace/config.json": '{"setting": "default"}',
                "/workspace/data/input.txt": "sample input data",
            },
            "messages_sent": [],
            "command_history": [],
            "search_results_cache": {},
            "current_directory": "/workspace",
            "available_tools": list(self.TOOLS.keys()),
            "target_artifact": "target_artifact",
            "inventory": [],
            "last_error": None,
            "action_count": 0,
            "tags": ["tool-environment", "realistic-tools"],
            "tool_version": "v2",
            "valid_sequence": ["read_file", "run_command", "write_file"],
            "next_index": 0,
        }

    def observe(self) -> dict[str, Any]:
        """Return current state observation."""
        obs = deepcopy(self._state)
        obs["available_tools"] = list(self.TOOLS.keys())
        return obs

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current state."""
        return deepcopy(self._state)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self._state = deepcopy(snapshot)

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool action and return result."""
        action_name = str(action.get("name", ""))
        params = action.get("params", {})

        if action_name not in self.TOOLS:
            self._state["last_error"] = f"Unknown tool: {action_name}"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        self._state["action_count"] += 1
        handler = getattr(self, f"_tool_{action_name}", None)

        if handler is None:
            self._state["last_error"] = f"Tool not implemented: {action_name}"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        try:
            result = handler(params)
            self._state["last_error"] = None
            return {"ok": True, "action": action_name, "observation": result}
        except Exception as e:
            self._state["last_error"] = str(e)
            return {"ok": False, "action": action_name, "error": str(e)}

    def _tool_read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read file content."""
        path = params.get("path", "")
        content = self._state["filesystem"].get(path)
        if content is None:
            raise ValueError(f"File not found: {path}")
        return {"path": path, "content": content}

    def _tool_write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Write file content."""
        path = params.get("path", "")
        content = params.get("content", "")
        self._state["filesystem"][path] = content
        return {"path": path, "bytes_written": len(content)}

    def _tool_search_web(self, params: dict[str, Any]) -> dict[str, Any]:
        """Search web (simulated)."""
        query = params.get("query", "")
        if query in self._state["search_results_cache"]:
            results = self._state["search_results_cache"][query]
        else:
            results = [
                {"title": f"Result 1 for '{query}'", "snippet": "Relevant information..."},
                {"title": f"Result 2 for '{query}'", "snippet": "More details..."},
            ]
            self._state["search_results_cache"][query] = results
        return {"query": query, "results": results}

    def _tool_run_command(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run command (simulated)."""
        command = params.get("command", "")
        self._state["command_history"].append(command)
        if "error" in command.lower():
            raise ValueError(f"Command failed: {command}")
        return {"command": command, "exit_code": 0, "output": f"Executed: {command}"}

    def _tool_send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send message."""
        recipient = params.get("recipient", "")
        message = params.get("message", "")
        self._state["messages_sent"].append({"recipient": recipient, "message": message})
        return {"recipient": recipient, "delivered": True}

    def _tool_list_files(self, params: dict[str, Any]) -> dict[str, Any]:
        """List files in directory."""
        directory = params.get("directory", "/workspace")
        files = [p for p in self._state["filesystem"].keys() if p.startswith(directory)]
        return {"directory": directory, "files": files}

    def _tool_delete_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Delete a file."""
        path = params.get("path", "")
        if path not in self._state["filesystem"]:
            raise ValueError(f"File not found: {path}")
        del self._state["filesystem"][path]
        return {"path": path, "deleted": True}

    def _tool_create_directory(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create directory (no-op in this simulation)."""
        path = params.get("path", "")
        return {"path": path, "created": True}

    @classmethod
    def clone_from_snapshot(
        cls, snapshot: dict[str, Any], *, seed: int = 7
    ) -> "ToolEnvironment":
        """Create a new environment from a snapshot."""
        env = cls(seed=seed)
        env.restore(snapshot)
        return env

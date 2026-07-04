"""Tests for ToolEnvironment."""


from smtr.runtime.tool_environment import ToolEnvironment


class TestToolEnvironmentBasics:
    """Test basic ToolEnvironment operations."""

    def test_observe_returns_state(self):
        """Test that observe() returns state dict."""
        env = ToolEnvironment(seed=7)
        obs = env.observe()

        assert isinstance(obs, dict)
        assert "filesystem" in obs
        assert "available_tools" in obs
        assert "target_artifact" in obs

    def test_snapshot_and_restore(self):
        """Test snapshot and restore roundtrip."""
        env = ToolEnvironment(seed=7)

        env.apply({"name": "read_file", "params": {"path": "/workspace/config.json"}})
        snapshot = env.snapshot()

        env2 = ToolEnvironment(seed=7)
        env2.restore(snapshot)

        assert env2.observe() == env.observe()

    def test_clone_from_snapshot(self):
        """Test clone_from_snapshot class method."""
        env = ToolEnvironment(seed=7)
        env.apply({"name": "read_file", "params": {"path": "/workspace/config.json"}})
        snapshot = env.snapshot()

        cloned = ToolEnvironment.clone_from_snapshot(snapshot, seed=42)

        assert cloned.observe() == env.observe()
        assert cloned.seed == 42


class TestToolEnvironmentTools:
    """Test individual tool operations."""

    def test_read_file_success(self):
        """Test reading existing file."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "read_file", "params": {"path": "/workspace/config.json"}})

        assert result["ok"] is True
        assert "content" in result["observation"]

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "read_file", "params": {"path": "/nonexistent"}})

        assert result["ok"] is False
        assert "error" in result

    def test_write_file(self):
        """Test writing file."""
        env = ToolEnvironment(seed=7)
        result = env.apply({
            "name": "write_file",
            "params": {"path": "/workspace/new.txt", "content": "hello"},
        })

        assert result["ok"] is True
        assert env._state["filesystem"]["/workspace/new.txt"] == "hello"

    def test_search_web(self):
        """Test web search."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "search_web", "params": {"query": "test"}})

        assert result["ok"] is True
        assert "results" in result["observation"]
        assert len(result["observation"]["results"]) > 0

    def test_search_web_caches_results(self):
        """Test that search results are cached."""
        env = ToolEnvironment(seed=7)
        env.apply({"name": "search_web", "params": {"query": "test"}})
        env.apply({"name": "search_web", "params": {"query": "test"}})

        assert len(env._state["search_results_cache"]["test"]) == 2

    def test_run_command_success(self):
        """Test running command."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "run_command", "params": {"command": "ls -la"}})

        assert result["ok"] is True
        assert "ls -la" in env._state["command_history"]

    def test_run_command_error(self):
        """Test running command with error."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "run_command", "params": {"command": "error command"}})

        assert result["ok"] is False
        assert "error" in result

    def test_send_message(self):
        """Test sending message."""
        env = ToolEnvironment(seed=7)
        result = env.apply({
            "name": "send_message",
            "params": {"recipient": "user", "message": "hello"},
        })

        assert result["ok"] is True
        assert len(env._state["messages_sent"]) == 1

    def test_list_files(self):
        """Test listing files."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "list_files", "params": {"directory": "/workspace"}})

        assert result["ok"] is True
        assert "files" in result["observation"]
        assert len(result["observation"]["files"]) > 0

    def test_delete_file(self):
        """Test deleting file."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "delete_file", "params": {"path": "/workspace/config.json"}})

        assert result["ok"] is True
        assert "/workspace/config.json" not in env._state["filesystem"]

    def test_delete_file_not_found(self):
        """Test deleting non-existent file."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "delete_file", "params": {"path": "/nonexistent"}})

        assert result["ok"] is False

    def test_create_directory(self):
        """Test creating directory."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "create_directory", "params": {"path": "/workspace/new_dir"}})

        assert result["ok"] is True


class TestToolEnvironmentEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_tool(self):
        """Test using unknown tool."""
        env = ToolEnvironment(seed=7)
        result = env.apply({"name": "unknown_tool"})

        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    def test_action_count_incremented(self):
        """Test that action count is incremented."""
        env = ToolEnvironment(seed=7)
        initial_count = env._state["action_count"]

        env.apply({"name": "read_file", "params": {"path": "/workspace/config.json"}})
        env.apply({"name": "search_web", "params": {"query": "test"}})

        assert env._state["action_count"] == initial_count + 2

    def test_last_error_cleared_on_success(self):
        """Test that last_error is cleared on success."""
        env = ToolEnvironment(seed=7)

        env.apply({"name": "read_file", "params": {"path": "/nonexistent"}})
        assert env._state["last_error"] is not None

        env.apply({"name": "read_file", "params": {"path": "/workspace/config.json"}})
        assert env._state["last_error"] is None

    def test_available_tools_listed(self):
        """Test that available tools are listed in observation."""
        env = ToolEnvironment(seed=7)
        obs = env.observe()

        assert "read_file" in obs["available_tools"]
        assert "write_file" in obs["available_tools"]
        assert "search_web" in obs["available_tools"]

"""Tests for the logging setup shared by verity-core and the tools."""

from __future__ import annotations

import io
import logging

from verity_core.logs import NAMESPACE, configure_logging, get_logger


class TestGetLogger:
    def test_returns_the_logger_for_a_module(self) -> None:
        assert get_logger("verity_core.corpus").name == "verity_core.corpus"

    def test_module_loggers_sit_under_the_namespace(self) -> None:
        # This is what lets a tool set one level for the whole library.
        assert get_logger("verity_core.batch").parent is logging.getLogger(NAMESPACE)

    def test_the_library_installs_a_placeholder_handler(self) -> None:
        # A library must not force handler choices on its host, but it also must not
        # trigger "no handlers could be found" warnings.
        import verity_core  # noqa: F401 - imported for its side effect

        handlers = logging.getLogger(NAMESPACE).handlers
        assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


class TestConfigureLogging:
    def test_writes_records_to_the_given_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(logging.INFO, stream=stream, force=True)
        get_logger("verity_core.test").info("something happened")
        assert "something happened" in stream.getvalue()

    def test_sets_the_level_on_the_namespace(self) -> None:
        configure_logging(logging.DEBUG, stream=io.StringIO(), force=True)
        assert logging.getLogger(NAMESPACE).level == logging.DEBUG

    def test_accepts_a_level_name(self) -> None:
        configure_logging("WARNING", stream=io.StringIO(), force=True)
        assert logging.getLogger(NAMESPACE).level == logging.WARNING

    def test_filters_records_below_the_level(self) -> None:
        stream = io.StringIO()
        configure_logging(logging.WARNING, stream=stream, force=True)
        logger = get_logger("verity_core.test")
        logger.debug("routine detail")
        logger.warning("recoverable problem")
        output = stream.getvalue()
        assert "routine detail" not in output
        assert "recoverable problem" in output

    def test_includes_the_level_and_logger_name(self) -> None:
        stream = io.StringIO()
        configure_logging(logging.INFO, stream=stream, force=True)
        get_logger("verity_core.runner").info("container started id=abc123")
        line = stream.getvalue()
        assert "INFO" in line
        assert "verity_core.runner" in line
        assert "container started id=abc123" in line

    def test_is_idempotent(self) -> None:
        stream = io.StringIO()
        configure_logging(logging.INFO, stream=stream, force=True)
        before = len(logging.getLogger(NAMESPACE).handlers)
        configure_logging(logging.INFO, stream=stream)
        configure_logging(logging.INFO, stream=stream)
        assert len(logging.getLogger(NAMESPACE).handlers) == before

    def test_does_not_duplicate_records(self) -> None:
        stream = io.StringIO()
        configure_logging(logging.INFO, stream=stream, force=True)
        configure_logging(logging.INFO, stream=stream)
        get_logger("verity_core.test").info("once")
        assert stream.getvalue().count("once") == 1

    def test_force_replaces_existing_handlers(self) -> None:
        first, second = io.StringIO(), io.StringIO()
        configure_logging(logging.INFO, stream=first, force=True)
        configure_logging(logging.INFO, stream=second, force=True)
        get_logger("verity_core.test").info("only in the second")
        assert first.getvalue() == ""
        assert "only in the second" in second.getvalue()

    def test_returns_the_namespace_logger(self) -> None:
        logger = configure_logging(logging.INFO, stream=io.StringIO(), force=True)
        assert logger is logging.getLogger(NAMESPACE)

    def test_leaves_propagation_alone_so_tools_can_capture_records(self) -> None:
        # Tools aggregate verity-core's records through their own root handlers.
        logging.getLogger(NAMESPACE).propagate = True
        configure_logging(logging.INFO, stream=io.StringIO(), force=True)
        assert logging.getLogger(NAMESPACE).propagate is True

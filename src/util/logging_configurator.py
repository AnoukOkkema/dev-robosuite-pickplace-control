import importlib.util
import logging
import logging.config
import os
from pathlib import Path

import yaml


class LoggingConfigurator:
    """Configures the application's logging system from config/logging.yaml."""

    _initialized = False

    @classmethod
    def setup(cls, log_filename: str = "main.log") -> logging.Logger:
        """
        Configures the logging system using the YAML configuration file.

        Only runs on the first call per process. Later calls just return
        the logger that was already configured.

        Args:
            log_filename (str): Name of the log file inside logs/, e.g.
                "train_yolo.log". This lets each entrypoint script write
                to its own file instead of sharing logs/main.log.

        Returns:
            logging.Logger: Configured logger instance.

        Example:
            logger = LoggingConfigurator.setup("train_yolo.log")
            logger.info("Application started")
        """

        if cls._initialized:
            return logging.getLogger(__name__)

        # Ensure the logs directory exists
        Path("logs").mkdir(exist_ok=True)

        # Load the YAML configuration
        config_path = Path("config", "logging.yaml")

        try:
            if config_path.exists():
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)

                    if "file" in config.get("handlers", {}):
                        config["handlers"]["file"]["filename"] = str(
                            Path("logs", log_filename)
                        )

                    logging.config.dictConfig(config)
            else:
                raise FileNotFoundError(f"Logging config not found at {config_path}")
        except Exception as e:
            print(f"Failed to load logging configuration: {e}")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | - %(message)s",
                datefmt="%H:%M:%S",
            )
        finally:
            cls._initialized = True

        return logging.getLogger(__name__)

    @staticmethod
    def suppress_robosuite_warnings() -> None:
        """
        Silences robosuite's own startup warnings (missing private macro
        file, optional mink-based whole-body IK controller). It does this
        by writing a private macro file that raises the console logging
        level to "ERROR", which is the same fix robosuite's own
        `scripts/setup_macros.py` applies manually.

        Must be called before robosuite is imported anywhere, whether
        directly or indirectly, since robosuite prints those warnings as
        a side effect of being imported. That means it can't be folded
        into `setup()` like the rest of the logging setup. It has to run
        at the top of each entrypoint, before any module that imports
        robosuite.

        Returns:
            None
        """

        spec = importlib.util.find_spec("robosuite")

        if spec is None or not spec.submodule_search_locations:
            return

        macros_private_path = os.path.join(
            spec.submodule_search_locations[0], "macros_private.py"
        )

        if os.path.exists(macros_private_path):
            return

        with open(macros_private_path, "w") as file:
            file.write('CONSOLE_LOGGING_LEVEL = "ERROR"\n')

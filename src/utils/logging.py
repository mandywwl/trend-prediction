import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


# TODO: Broaden logging configuration for structured output.

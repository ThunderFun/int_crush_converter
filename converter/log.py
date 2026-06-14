"""Shared logging configuration for the converter package.

Usage:
    from converter.log import logger
    logger.info("message")

CLI controls the log level:
    --verbose  → DEBUG  (algorithm internals, per-layer metrics)
    (default)  → INFO   (progress, warnings)
    --quiet    → WARNING (only problems)

Library users: the package logger has a ``NullHandler`` so that importing
``converter`` never emits messages unless the caller explicitly configures
a handler on the ``converter`` logger.  This follows the standard library
logging pattern recommended in the docs.
"""

import logging

logger = logging.getLogger("converter")
logger.addHandler(logging.NullHandler())

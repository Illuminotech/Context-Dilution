"""Event notification handlers using the observer pattern."""

from __future__ import annotations

import logging
from typing import Any

from models.stock import Order, Product

logger = logging.getLogger(__name__)


class LoggingObserver:
    """Logs order events for audit trail."""

    def on_order_created(self, order: Order) -> None:
        logger.info("Order %d created for customer %d", order.id, order.customer_id)

    def on_order_fulfilled(self, order: Order) -> None:
        logger.info("Order %d fulfilled", order.id)

    def on_stock_low(self, product: Product) -> None:
        logger.warning(
            "Low stock alert: %s has %d units (reorder at %d)",
            product.sku,
            product.stock_quantity,
            product.reorder_level,
        )


class AlertObserver:
    """Sends alerts for critical inventory events."""

    def __init__(self, alert_threshold: int = 5) -> None:
        self._threshold = alert_threshold

    def on_order_created(self, order: Order) -> None:
        pass  # no alert for new orders

    def on_order_fulfilled(self, order: Order) -> None:
        pass  # no alert for fulfillments

    def on_stock_low(self, product: Product) -> None:
        if product.stock_quantity <= self._threshold:
            logger.critical(
                "CRITICAL: %s stock at %d — immediate reorder required",
                product.sku,
                product.stock_quantity,
            )

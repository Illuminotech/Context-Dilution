"""Transaction service — processes orders with stock management.

Uses the observer pattern for event notification.
NO caching in this layer — previous caching caused race conditions.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from models.stock import Customer, Order, OrderItem, Product


class OrderEventObserver(Protocol):
    """Observer protocol for order lifecycle events."""

    def on_order_created(self, order: Order) -> None: ...
    def on_order_fulfilled(self, order: Order) -> None: ...
    def on_stock_low(self, product: Product) -> None: ...


class TransactionService:
    """Handles order processing and stock management."""

    def __init__(self, session: Session, observers: list[OrderEventObserver] | None = None) -> None:
        self._session = session
        self._observers = observers or []

    def add_observer(self, observer: OrderEventObserver) -> None:
        self._observers.append(observer)

    def _notify(self, event: str, **kwargs: object) -> None:
        for obs in self._observers:
            handler = getattr(obs, event, None)
            if handler:
                handler(**kwargs)

    def create_order(self, customer: Customer, items: list[dict[str, int | float]]) -> Order:
        """Create a new order with stock validation.

        Args:
            customer: The Customer placing the order.
            items: List of dicts with 'product_id' and 'quantity'.
        """
        order = Order(customer_id=customer.id, status="pending")
        self._session.add(order)
        self._session.flush()  # get order.id

        for item_data in items:
            product = self._session.get(Product, item_data["product_id"])
            if product is None:
                raise ValueError(f"Product not found: {item_data['product_id']}")

            quantity = int(item_data["quantity"])
            product.adjust_stock(-quantity)

            order_item = OrderItem(
                order_id=order.id,
                product_id=product.id,
                quantity=quantity,
                unit_price=product.price,
            )
            self._session.add(order_item)

            if product.is_low_stock():
                self._notify("on_stock_low", product=product)

        self._session.flush()
        self._notify("on_order_created", order=order)
        return order

    def fulfill_order(self, order_id: int) -> Order:
        """Mark an order as fulfilled."""
        order = self._session.get(Order, order_id)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        if order.status != "pending":
            raise ValueError(f"Cannot fulfill order {order_id} with status '{order.status}'")

        order.status = "fulfilled"
        self._session.flush()
        self._notify("on_order_fulfilled", order=order)
        return order

    def cancel_order(self, order_id: int) -> Order:
        """Cancel an order and restore stock."""
        order = self._session.get(Order, order_id)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        if order.status not in ("pending",):
            raise ValueError(f"Cannot cancel order {order_id} with status '{order.status}'")

        for item in order.items:
            item.product.adjust_stock(item.quantity)

        order.status = "cancelled"
        self._session.flush()
        return order

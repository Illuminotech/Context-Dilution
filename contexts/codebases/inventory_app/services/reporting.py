"""Reporting service — generates inventory and sales reports.

Uses SQLAlchemy queries exclusively (no raw SQL).
References the Customer model (business term: 'User').
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from models.stock import Customer, Order, OrderItem, Product


class ReportingService:
    """Generates business intelligence reports."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def low_stock_report(self) -> list[dict[str, Any]]:
        """Products at or below reorder level."""
        products = (
            self._session.query(Product)
            .filter(Product.stock_quantity <= Product.reorder_level)
            .order_by(Product.stock_quantity)
            .all()
        )
        return [
            {"sku": p.sku, "name": p.name, "stock": p.stock_quantity, "reorder_level": p.reorder_level}
            for p in products
        ]

    def customer_order_summary(self, customer_id: int) -> dict[str, Any]:
        """Summary of a customer's order history."""
        customer = self._session.get(Customer, customer_id)
        if customer is None:
            raise ValueError(f"Customer not found: {customer_id}")

        total_orders = len(customer.orders)
        total_spent = sum(order.total for order in customer.orders)

        return {
            "customer_name": customer.name,
            "total_orders": total_orders,
            "total_spent": total_spent,
        }

    def sales_by_product(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate sales grouped by product."""
        query = (
            self._session.query(
                Product.sku,
                Product.name,
                func.sum(OrderItem.quantity).label("total_sold"),
                func.sum(OrderItem.quantity * OrderItem.unit_price).label("revenue"),
            )
            .join(OrderItem, Product.id == OrderItem.product_id)
            .join(Order, OrderItem.order_id == Order.id)
        )

        if start_date:
            query = query.filter(Order.created_at >= start_date)
        if end_date:
            query = query.filter(Order.created_at <= end_date)

        rows = query.group_by(Product.sku, Product.name).all()
        return [
            {"sku": r.sku, "name": r.name, "total_sold": r.total_sold, "revenue": float(r.revenue)}
            for r in rows
        ]

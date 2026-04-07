"""Stock and inventory models using SQLAlchemy ORM."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    """Customer entity — referred to as 'User' in business discussions."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    orders = relationship("Order", back_populates="customer")

    def __repr__(self) -> str:
        return f"Customer(id={self.id}, name={self.name})"


class Product(Base):
    """Product with stock tracking."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    sku = Column(String(50), unique=True, nullable=False)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    reorder_level = Column(Integer, default=10)

    order_items = relationship("OrderItem", back_populates="product")

    def is_low_stock(self) -> bool:
        return self.stock_quantity <= self.reorder_level

    def adjust_stock(self, delta: int) -> None:
        """Adjust stock by delta (positive=restock, negative=sold)."""
        new_qty = self.stock_quantity + delta
        if new_qty < 0:
            raise ValueError(f"Insufficient stock for {self.sku}: have {self.stock_quantity}, need {-delta}")
        self.stock_quantity = new_qty

    def __repr__(self) -> str:
        return f"Product(sku={self.sku}, stock={self.stock_quantity})"


class Order(Base):
    """Customer order."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

    @property
    def total(self) -> float:
        return sum(item.subtotal for item in self.items)

    def __repr__(self) -> str:
        return f"Order(id={self.id}, status={self.status})"


class OrderItem(Base):
    """Line item in an order."""

    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")

    @property
    def subtotal(self) -> float:
        return self.quantity * self.unit_price

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
InvoiceReconcileEnv environment implementation.

The OpenEnv HTTP wrapper creates a fresh environment object on each /reset and
/step request, so this environment stores episode state at the class level to
preserve HTTP episode continuity. Rewards and final scores are kept strictly
inside (0, 1) for Phase 2 validation.
"""

from __future__ import annotations

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, InvoicereconcileenvAction, InvoicereconcileenvObservation
except ImportError:
    from models import ActionType, InvoicereconcileenvAction, InvoicereconcileenvObservation


_OCR_VARIANTS = {
    "Industrial Widget": [
        "Industrial Widget",
        "INDUSTRIAL WIDGET",
        "Industral Widget",
        "Industrial Wdget",
        "WIDGET-IND-2025",
        "lndustrial Widget",
    ],
    "Acme Supplies": ["Acme Supplies", "ACME SUPPLIES", "Acme Suppies", "Acm3 Supplies"],
    "Global Parts Co": ["Global Parts Co", "GLOBAL PARTS CO", "G1obal Parts Co", "Global Part Co"],
    "FastShip Ltd": ["FastShip Ltd", "FASTSHIP LTD", "FastSh1p Ltd", "Fast Ship Ltd"],
}

TOLERANCE_SOFT = 0.02
TOLERANCE_HARD = 0.05
DEFAULT_REWARD = 0.5
MIN_SCORE = 0.001
MAX_SCORE = 0.999
MAX_STEPS = 40


def _bounded_unit(value: float, *, digits: int = 6) -> float:
    if value != value:
        value = DEFAULT_REWARD
    if value <= 0.0:
        value = MIN_SCORE
    elif value >= 1.0:
        value = MAX_SCORE
    value = round(value, digits)
    if value <= 0.0:
        return MIN_SCORE
    if value >= 1.0:
        return MAX_SCORE
    return value


def _step_reward(value: float) -> float:
    return _bounded_unit(value, digits=3)


def apply_ocr_noise(text: str, rng: random.Random, level: str = "hard") -> str:
    if level != "hard":
        return text
    variants = _OCR_VARIANTS.get(text)
    if variants:
        return rng.choice(variants)
    if text and rng.random() < 0.3:
        chars = list(text)
        idx = rng.randint(0, len(chars) - 1)
        chars[idx] = rng.choice("01lI")
        return "".join(chars)
    return text


def generate_scenario(task_level: str, seed: int = 42):
    rng = random.Random(seed)

    vendors = [
        {"id": "V001", "name": "Acme Supplies", "bank_account": "BANK-ACC-001"},
        {"id": "V002", "name": "Global Parts Co", "bank_account": "BANK-ACC-002"},
        {"id": "V003", "name": "FastShip Ltd", "bank_account": "BANK-ACC-003"},
    ]

    def make_invoice(
        inv_id: str,
        vendor: dict[str, Any],
        qty: int,
        unit_price: float,
        *,
        ocr: bool = False,
        noise_level: str = "easy",
        priority: bool = False,
        discount_pct: float = 0.0,
        discount_steps: int = 5,
    ) -> dict[str, Any]:
        desc = apply_ocr_noise("Industrial Widget", rng, noise_level) if ocr else "Industrial Widget"
        v_name = apply_ocr_noise(vendor["name"], rng, noise_level) if ocr else vendor["name"]
        subtotal = round(qty * unit_price, 2)
        tax = round(subtotal * 0.18, 2)
        return {
            "invoice_id": inv_id,
            "vendor_id": vendor["id"],
            "vendor_name": v_name,
            "line_items": [{"description": desc, "quantity": qty, "unit_price": unit_price}],
            "subtotal": subtotal,
            "tax": tax,
            "total": round(subtotal + tax, 2),
            "po_reference": f"PO-{inv_id}",
            "bank_account": vendor["bank_account"],
            "priority": priority,
            "early_payment_discount_pct": discount_pct,
            "discount_deadline_steps": discount_steps,
        }

    def make_po(inv_id: str, vendor: dict[str, Any], qty: int, unit_price: float) -> dict[str, Any]:
        total = round(qty * unit_price, 2)
        return {
            "po_id": f"PO-{inv_id}",
            "vendor_id": vendor["id"],
            "approved_qty": qty,
            "agreed_unit_price": unit_price,
            "total_value": total,
            "bank_account": vendor["bank_account"],
        }

    def make_receipt(inv_id: str, qty: int) -> dict[str, Any]:
        return {
            "receipt_id": f"GR-{inv_id}",
            "po_reference": f"PO-{inv_id}",
            "received_qty": qty,
        }

    if task_level == "easy":
        vendor = vendors[0]
        qty, price = 100, 25.00
        inv = make_invoice("INV-001", vendor, qty, price)
        po = make_po("INV-001", vendor, qty, price)
        receipt = make_receipt("INV-001", qty)
        ground_truth = {
            "INV-001": {
                "correct_action": "approve",
                "has_discrepancy": False,
                "discrepancy_type": None,
                "correct_amount": inv["total"],
                "price_variance_pct": 0.01,
            }
        }
        return [inv], {"PO-INV-001": po}, {"GR-INV-001": receipt}, ground_truth

    if task_level == "medium":
        invoices, pos, receipts, ground_truth = [], {}, {}, {}

        v = vendors[0]
        qty, price = 50, 10.00
        inv = make_invoice("INV-101", v, qty, price)
        invoices.append(inv)
        pos["PO-INV-101"] = make_po("INV-101", v, qty, price)
        receipts["GR-INV-101"] = make_receipt("INV-101", qty)
        ground_truth["INV-101"] = {
            "correct_action": "approve",
            "has_discrepancy": False,
            "discrepancy_type": None,
            "correct_amount": inv["total"],
            "price_variance_pct": 0.01,
        }

        v = vendors[1]
        qty = 30
        agreed_price, invoice_price = 20.00, 25.00
        variance_pct = abs(invoice_price - agreed_price) / agreed_price
        inv2 = make_invoice("INV-102", v, qty, invoice_price)
        invoices.append(inv2)
        pos["PO-INV-102"] = make_po("INV-102", v, qty, agreed_price)
        receipts["GR-INV-102"] = make_receipt("INV-102", qty)
        ground_truth["INV-102"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "price",
            "correct_amount": None,
            "price_variance_pct": round(variance_pct, 4),
        }

        v = vendors[2]
        qty_invoiced, qty_received, price = 100, 80, 5.00
        inv3 = make_invoice("INV-103", v, qty_invoiced, price)
        invoices.append(inv3)
        pos["PO-INV-103"] = make_po("INV-103", v, qty_invoiced, price)
        receipts["GR-INV-103"] = make_receipt("INV-103", qty_received)
        ground_truth["INV-103"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "quantity",
            "correct_amount": None,
            "price_variance_pct": 0.01,
        }

        return invoices, pos, receipts, ground_truth

    invoices, pos, receipts, ground_truth = [], {}, {}, {}

    v = vendors[0]
    qty, price = 200, 8.50
    inv = make_invoice("INV-201", v, qty, price, ocr=True, noise_level="hard")
    invoices.append(inv)
    pos["PO-INV-201"] = make_po("INV-201", v, qty, price)
    receipts["GR-INV-201"] = make_receipt("INV-201", qty)
    ground_truth["INV-201"] = {
        "correct_action": "approve",
        "has_discrepancy": False,
        "discrepancy_type": None,
        "correct_amount": inv["total"],
        "price_variance_pct": 0.01,
    }

    v = vendors[1]
    qty = 100
    agreed_price, invoice_price = 15.00, 15.25
    variance_pct = abs(invoice_price - agreed_price) / agreed_price
    inv2 = make_invoice("INV-202", v, qty, invoice_price, ocr=True, noise_level="hard")
    invoices.append(inv2)
    pos["PO-INV-202"] = make_po("INV-202", v, qty, agreed_price)
    receipts["GR-INV-202"] = make_receipt("INV-202", qty)
    ground_truth["INV-202"] = {
        "correct_action": "approve",
        "has_discrepancy": False,
        "discrepancy_type": None,
        "correct_amount": inv2["total"],
        "price_variance_pct": round(variance_pct, 4),
    }

    dup_price = 8.55
    dup_variance_pct = abs(dup_price - price) / price
    inv3 = make_invoice("INV-203", vendors[0], 200, dup_price, ocr=True, noise_level="hard")
    inv3["po_reference"] = "PO-INV-201"
    invoices.append(inv3)
    pos["PO-INV-203"] = make_po("INV-201", vendors[0], 200, price)
    receipts["GR-INV-203"] = make_receipt("INV-203", 200)
    ground_truth["INV-203"] = {
        "correct_action": "flag",
        "has_discrepancy": True,
        "discrepancy_type": "duplicate",
        "correct_amount": None,
        "price_variance_pct": round(dup_variance_pct, 4),
    }

    v = vendors[2]
    inv4 = make_invoice(
        "INV-204",
        v,
        qty=100,
        unit_price=30.00,
        ocr=True,
        noise_level="hard",
        priority=True,
        discount_pct=0.02,
        discount_steps=4,
    )
    invoices.append(inv4)
    pos["PO-INV-204"] = make_po("INV-204", v, qty=100, unit_price=30.00)
    receipts["GR-INV-204"] = make_receipt("INV-204", qty=60)
    ground_truth["INV-204"] = {
        "correct_action": "flag",
        "has_discrepancy": True,
        "discrepancy_type": "quantity",
        "correct_amount": None,
        "price_variance_pct": 0.01,
        "priority": True,
        "discount_pct": 0.02,
    }

    fraud_vendor = {"id": "V003", "name": "FastShip Ltd", "bank_account": "BANK-ACC-FRAUD-999"}
    inv5 = make_invoice("INV-205", fraud_vendor, qty=50, unit_price=30.00, ocr=True, noise_level="hard")
    inv5["bank_account"] = "BANK-ACC-FRAUD-999"
    invoices.append(inv5)
    pos["PO-INV-205"] = make_po("INV-205", vendors[2], 50, 30.00)
    receipts["GR-INV-205"] = make_receipt("INV-205", 50)
    ground_truth["INV-205"] = {
        "correct_action": "escalate",
        "has_discrepancy": True,
        "discrepancy_type": "vendor",
        "correct_amount": None,
        "price_variance_pct": 0.01,
    }

    return invoices, pos, receipts, ground_truth


def grade_episode(
    ground_truth: dict[str, dict[str, Any]],
    decisions: dict[str, str],
    flags: dict[str, str],
    steps_taken: int,
    max_steps: int,
    priority_bonuses: Optional[dict[str, dict[str, Any]]] = None,
) -> float:
    if not ground_truth:
        return _bounded_unit(DEFAULT_REWARD)

    priority_bonuses = priority_bonuses or {}
    score = 0.0
    per_invoice = 1.0 / len(ground_truth)

    for inv_id, truth in ground_truth.items():
        decision = decisions.get(inv_id, "none")
        flagged_type = flags.get(inv_id)
        variance_pct = truth.get("price_variance_pct", 0.01)
        correct_action = truth.get("correct_action")

        if correct_action == "approve":
            if decision == "approve":
                score += per_invoice
            elif decision == "flag":
                if variance_pct <= TOLERANCE_SOFT:
                    score += per_invoice * 0.2
                elif variance_pct <= TOLERANCE_HARD:
                    score += per_invoice * 0.5
                else:
                    score += per_invoice * 0.01
            else:
                score += per_invoice * 0.01
            continue

        if correct_action == "flag":
            if decision == "flag":
                score += per_invoice if flagged_type == truth.get("discrepancy_type") else per_invoice * 0.5
            elif decision == "approve":
                if truth.get("discrepancy_type") == "price" and variance_pct <= TOLERANCE_HARD:
                    score += per_invoice * 0.3
                else:
                    score += per_invoice * 0.01
            else:
                score += per_invoice * 0.01
            continue

        if correct_action == "escalate":
            if decision == "escalate":
                score += per_invoice
            elif decision == "flag":
                score += per_invoice * 0.4
            else:
                score += per_invoice * 0.01
            continue

        if correct_action == "reject":
            score += per_invoice if decision == "reject" else per_invoice * 0.01

    if max_steps > 0 and (steps_taken / max_steps) > 0.80:
        score *= 0.85

    for inv_id, bonus_info in priority_bonuses.items():
        truth = ground_truth.get(inv_id, {})
        if truth.get("correct_action") == "approve" and decisions.get(inv_id) == "approve" and bonus_info.get("captured"):
            score += 0.05

    return _bounded_unit(score)


class InvoicereconcileenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False
    MAX_STEPS = MAX_STEPS

    _episode_id = str(uuid4())
    _task_level = "easy"
    _step_count = 0
    _current_index = 0
    _invoices: list[dict[str, Any]] = []
    _pos: dict[str, dict[str, Any]] = {}
    _receipts: dict[str, dict[str, Any]] = {}
    _ground_truth: dict[str, dict[str, Any]] = {}
    _decisions: dict[str, str] = {}
    _flags: dict[str, str] = {}
    _extracted: dict[str, bool] = {}
    _po_retrieved: dict[str, bool] = {}
    _receipt_retrieved: dict[str, bool] = {}
    _batch_status: dict[str, str] = {}
    _priority_bonuses: dict[str, dict[str, Any]] = {}
    _cumulative_reward = 0.0
    _final_score: Optional[float] = None

    def __init__(self):
        super().__init__()
        cls = type(self)
        if not cls._invoices:
            self._reset_shared_state(task_level="easy", seed=42, episode_id=cls._episode_id)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> InvoicereconcileenvObservation:
        options = options or {}
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            options = {**kwargs["options"], **options}

        task_level = kwargs.get("task_level", options.get("task_level", "easy"))
        scenario_seed = kwargs.get("scenario_seed", options.get("seed", seed if seed is not None else 42))
        self._reset_shared_state(task_level=task_level, seed=int(scenario_seed), episode_id=episode_id)

        return self._build_observation(
            message=f"Episode started. Task: {task_level}. {len(type(self)._invoices)} invoice(s) to process.",
            reward=None,
            done=False,
            po_data=None,
            receipt_data=None,
        )

    def step(
        self,
        action: InvoicereconcileenvAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoicereconcileenvObservation:
        del timeout_s, kwargs
        cls = type(self)
        cls._step_count += 1
        reward = 0.02
        done = False
        message = ""

        current_inv = self._current_invoice()
        inv_id = current_inv["invoice_id"] if current_inv else None

        if action.action_type == ActionType.EXTRACT_FIELDS:
            if current_inv and inv_id not in cls._extracted:
                cls._extracted[inv_id] = True
                reward = 0.04
                priority_note = ""
                if current_inv.get("priority"):
                    remaining = current_inv.get("discount_deadline_steps", 0)
                    discount = current_inv.get("early_payment_discount_pct", 0.0) * 100
                    priority_note = f" PRIORITY: {discount:.1f}% early-payment discount expires in {remaining} steps."
                message = (
                    f"Extracted fields for {inv_id}. Vendor: {current_inv['vendor_name']} "
                    f"(ID: {current_inv['vendor_id']}), Total: {current_inv['total']}, "
                    f"PO Ref: {current_inv['po_reference']}, Bank: {current_inv['bank_account']}."
                    f"{priority_note}"
                )
            else:
                message = "Already extracted or no invoice available."

        elif action.action_type == ActionType.RETRIEVE_PO:
            po_key = f"PO-{inv_id}"
            if current_inv and po_key in cls._pos and inv_id not in cls._po_retrieved:
                cls._po_retrieved[inv_id] = True
                reward = 0.03
                po = cls._pos[po_key]
                message = (
                    f"PO retrieved for {inv_id}: agreed qty={po['approved_qty']}, "
                    f"agreed price={po['agreed_unit_price']}, vendor_id={po['vendor_id']}, "
                    f"bank={po.get('bank_account', 'N/A')}."
                )
            else:
                message = "PO not found or already retrieved."

        elif action.action_type == ActionType.RETRIEVE_RECEIPT:
            gr_key = f"GR-{inv_id}"
            if current_inv and gr_key in cls._receipts and inv_id not in cls._receipt_retrieved:
                cls._receipt_retrieved[inv_id] = True
                reward = 0.03
                receipt = cls._receipts[gr_key]
                invoiced_qty = current_inv["line_items"][0]["quantity"] if current_inv.get("line_items") else 0
                received_qty = receipt["received_qty"]
                partial_note = ""
                if received_qty < invoiced_qty:
                    partial_note = (
                        f" PARTIAL SHIPMENT: invoice claims {invoiced_qty} units but only {received_qty} received."
                    )
                message = f"Goods receipt for {inv_id}: received qty={received_qty}.{partial_note}"
            else:
                message = "Receipt not found or already retrieved."

        elif action.action_type == ActionType.FLAG_DISCREPANCY:
            reward, message = self._handle_terminal_decision(inv_id, "flag", action)
        elif action.action_type == ActionType.APPROVE_PAYMENT:
            reward, message = self._handle_terminal_decision(inv_id, "approve", action)
        elif action.action_type == ActionType.REJECT_INVOICE:
            reward, message = self._handle_terminal_decision(inv_id, "reject", action)
        elif action.action_type == ActionType.ESCALATE:
            reward, message = self._handle_terminal_decision(inv_id, "escalate", action)
        else:
            message = f"Unsupported action '{action.action_type}'."

        all_decided = len(cls._decisions) == len(cls._invoices)
        max_steps_hit = cls._step_count >= cls.MAX_STEPS

        if all_decided or max_steps_hit:
            cls._final_score = grade_episode(
                cls._ground_truth,
                cls._decisions,
                cls._flags,
                cls._step_count,
                cls.MAX_STEPS,
                priority_bonuses=cls._priority_bonuses,
            )
            reward = _step_reward(reward + (cls._final_score * 0.04))
            done = True
            message = (
                f"{message} | EPISODE COMPLETE. Final grade: {cls._final_score:.6f}. "
                f"Decisions: {cls._decisions}."
            ).strip()
        else:
            reward = _step_reward(reward)

        cls._cumulative_reward = _bounded_unit(cls._cumulative_reward + reward)
        po_data = cls._pos.get(f"PO-{inv_id}") if inv_id and inv_id in cls._po_retrieved else None
        receipt_data = cls._receipts.get(f"GR-{inv_id}") if inv_id and inv_id in cls._receipt_retrieved else None
        return self._build_observation(
            message=message,
            reward=reward,
            done=done,
            po_data=po_data,
            receipt_data=receipt_data,
        )

    @property
    def state(self) -> State:
        cls = type(self)
        return State(
            episode_id=cls._episode_id,
            step_count=cls._step_count,
            current_invoice_index=cls._current_index,
            cumulative_reward=cls._cumulative_reward,
            final_score=cls._final_score,
        )

    def _reset_shared_state(self, *, task_level: str, seed: int, episode_id: Optional[str]) -> None:
        cls = type(self)
        cls._episode_id = episode_id or str(uuid4())
        cls._task_level = task_level
        cls._step_count = 0
        cls._current_index = 0
        cls._invoices, cls._pos, cls._receipts, cls._ground_truth = generate_scenario(task_level, seed)
        cls._decisions = {}
        cls._flags = {}
        cls._extracted = {}
        cls._po_retrieved = {}
        cls._receipt_retrieved = {}
        cls._batch_status = {inv["invoice_id"]: "pending" for inv in cls._invoices}
        cls._priority_bonuses = {}
        cls._cumulative_reward = 0.0
        cls._final_score = None

    def _current_invoice(self) -> Optional[dict[str, Any]]:
        cls = type(self)
        if 0 <= cls._current_index < len(cls._invoices):
            return cls._invoices[cls._current_index]
        return None

    def _advance_invoice(self) -> None:
        cls = type(self)
        if cls._current_index < len(cls._invoices):
            cls._current_index += 1

    def _steps_used_on_invoice(self, inv_id: str) -> int:
        cls = type(self)
        return sum(
            1
            for seen in (
                cls._extracted.get(inv_id),
                cls._po_retrieved.get(inv_id),
                cls._receipt_retrieved.get(inv_id),
            )
            if seen
        ) + 1

    def _handle_terminal_decision(
        self,
        inv_id: Optional[str],
        decision: str,
        action: InvoicereconcileenvAction,
    ) -> tuple[float, str]:
        cls = type(self)
        if not inv_id or inv_id in cls._decisions:
            return 0.02, "Already decided on this invoice."

        truth = cls._ground_truth.get(inv_id, {})
        cls._decisions[inv_id] = decision

        if decision == "flag":
            flagged_as = action.discrepancy_type.value if action.discrepancy_type else "other"
            cls._flags[inv_id] = flagged_as
            cls._batch_status[inv_id] = "flagged"
            if truth.get("has_discrepancy"):
                expected_type = truth.get("discrepancy_type")
                if flagged_as == expected_type:
                    reward = 0.08
                    message = f"Correct: {inv_id} flagged as '{flagged_as}'."
                else:
                    reward = 0.04
                    message = f"Discrepancy flagged but wrong type for {inv_id}. Expected '{expected_type}', got '{flagged_as}'."
            else:
                variance_pct = truth.get("price_variance_pct", 0.01)
                reward = 0.02
                if variance_pct <= TOLERANCE_SOFT:
                    message = (
                        f"False flag: {inv_id} was within tolerance "
                        f"({variance_pct * 100:.2f}% <= {TOLERANCE_SOFT * 100:.0f}%). Should approve."
                    )
                elif variance_pct <= TOLERANCE_HARD:
                    message = (
                        f"Cautious flag on {inv_id}: price variance {variance_pct * 100:.2f}% is in the grey zone "
                        f"({TOLERANCE_SOFT * 100:.0f}% to {TOLERANCE_HARD * 100:.0f}%)."
                    )
                else:
                    message = f"False flag: {inv_id} had no discrepancy."
            self._advance_invoice()
            return reward, message

        if decision == "approve":
            cls._batch_status[inv_id] = "approved"
            if truth.get("correct_action") == "approve":
                reward = 0.09
                message = f"{inv_id} correctly approved. Amount: {truth.get('correct_amount')}."
                current_inv = next((inv for inv in cls._invoices if inv["invoice_id"] == inv_id), None)
                if current_inv and current_inv.get("priority"):
                    deadline = current_inv.get("discount_deadline_steps", 0)
                    if self._steps_used_on_invoice(inv_id) <= deadline:
                        discount = current_inv.get("early_payment_discount_pct", 0.0)
                        bonus_amt = round(truth.get("correct_amount", 0.0) * discount, 2)
                        reward += 0.03
                        message += f" Early payment discount captured. Savings: {bonus_amt}."
                        cls._priority_bonuses[inv_id] = {"captured": True}
                    else:
                        message += " Discount window missed due to step count."
                        cls._priority_bonuses[inv_id] = {"captured": False}
            else:
                variance_pct = truth.get("price_variance_pct", 0.01)
                reward = 0.03
                if truth.get("discrepancy_type") == "price" and variance_pct <= TOLERANCE_HARD:
                    message = (
                        f"Questionable approval of {inv_id}: price variance {variance_pct * 100:.2f}% "
                        f"is above soft tolerance. Expected '{truth.get('correct_action')}'."
                    )
                else:
                    message = f"Wrong approval of {inv_id}. Expected '{truth.get('correct_action')}'."
            self._advance_invoice()
            return reward, message

        if decision == "reject":
            cls._batch_status[inv_id] = "rejected"
            reward = 0.06 if truth.get("correct_action") == "reject" else 0.02
            message = (
                f"{inv_id} correctly rejected."
                if truth.get("correct_action") == "reject"
                else f"Rejected {inv_id} but expected '{truth.get('correct_action')}'."
            )
            self._advance_invoice()
            return reward, message

        cls._batch_status[inv_id] = "escalated"
        reward = 0.08 if truth.get("correct_action") == "escalate" else 0.02
        message = (
            f"{inv_id} correctly escalated. Reason: {action.reason or 'No reason provided'}."
            if truth.get("correct_action") == "escalate"
            else f"Escalated {inv_id} unnecessarily. Expected '{truth.get('correct_action')}'."
        )
        self._advance_invoice()
        return reward, message

    def _build_observation(
        self,
        *,
        message: str,
        reward: Optional[float],
        done: bool,
        po_data: Optional[dict[str, Any]],
        receipt_data: Optional[dict[str, Any]],
    ) -> InvoicereconcileenvObservation:
        cls = type(self)
        current_inv = self._current_invoice()
        return InvoicereconcileenvObservation(
            message=message,
            current_invoice=_serialize_invoice(current_inv) if current_inv else None,
            po_data=po_data,
            receipt_data=receipt_data,
            flags=list(cls._flags.values()),
            batch_status=dict(cls._batch_status),
            step_count=cls._step_count,
            task_level=cls._task_level,
            done=done,
            reward=reward,
            priority_invoice_active=bool(current_inv and current_inv.get("priority")),
            discount_deadline_steps=current_inv.get("discount_deadline_steps", 0) if current_inv else 0,
            early_payment_discount_pct=current_inv.get("early_payment_discount_pct", 0.0) if current_inv else 0.0,
        )


def _serialize_invoice(inv: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not inv:
        return {}
    return {
        "invoice_id": inv.get("invoice_id"),
        "vendor_id": inv.get("vendor_id"),
        "vendor_name": inv.get("vendor_name"),
        "total": inv.get("total"),
        "po_reference": inv.get("po_reference"),
        "line_items": inv.get("line_items", []),
        "bank_account": inv.get("bank_account"),
        "priority": inv.get("priority", False),
        "early_payment_discount_pct": inv.get("early_payment_discount_pct", 0.0),
        "discount_deadline_steps": inv.get("discount_deadline_steps", 0),
    }

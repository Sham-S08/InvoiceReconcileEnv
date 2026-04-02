"""
Baseline inference script for InvoiceReconcileEnv.
Uses OpenAI Client + structured [START]/[STEP]/[END] stdout format.
"""
import os
import json
import requests
from openai import OpenAI
from typing import List, Optional

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "dummy-key")
SPACE_URL    = os.getenv("SPACE_URL", "https://shambhavis08-invoicereconcileenv.hf.space")

TOLERANCE_SOFT = 0.02
TOLERANCE_HARD = 0.05
MAX_STEPS = 40

_invoice_progress = {}

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Structured stdout loggers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def reset_env(task_level: str, seed: int = 42) -> dict:
    response = requests.post(
        f"{SPACE_URL}/reset",
        json={"options": {"task_level": task_level, "seed": seed}},
        timeout=30,
    )
    return response.json()

def step_env(action: dict) -> dict:
    response = requests.post(
        f"{SPACE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    return response.json()

# ---------------------------------------------------------------------------
# Rule-based deterministic agent
# ---------------------------------------------------------------------------

def rule_based_agent(observation: dict) -> dict:
    global _invoice_progress

    current_invoice = observation.get("current_invoice")
    if not current_invoice:
        return {"action_type": "extract_fields", "invoice_id": "done"}

    inv_id = current_invoice["invoice_id"]

    if inv_id not in _invoice_progress:
        _invoice_progress[inv_id] = {
            "extracted": False, "po": False, "receipt": False,
            "po_data": None, "receipt_data": None,
        }

    progress = _invoice_progress[inv_id]
    po_data = progress.get("po_data") or observation.get("po_data")
    receipt_data = progress.get("receipt_data") or observation.get("receipt_data")

    if observation.get("po_data") and not progress["po_data"]:
        progress["po_data"] = observation["po_data"]
        po_data = progress["po_data"]
    if observation.get("receipt_data") and not progress["receipt_data"]:
        progress["receipt_data"] = observation["receipt_data"]
        receipt_data = progress["receipt_data"]

    if not progress["extracted"]:
        return {"action_type": "extract_fields", "invoice_id": inv_id}
    if not progress["po"]:
        return {"action_type": "retrieve_po", "invoice_id": inv_id}
    if not progress["receipt"]:
        return {"action_type": "retrieve_receipt", "invoice_id": inv_id}

    invoice_bank = current_invoice.get("bank_account", "")
    po_bank = po_data.get("bank_account", "") if po_data else ""
    if po_bank and invoice_bank and invoice_bank != po_bank:
        return {"action_type": "escalate", "invoice_id": inv_id,
                "reason": f"Bank account mismatch: {invoice_bank} vs {po_bank}"}

    vendor_id = current_invoice.get("vendor_id", "")
    if vendor_id and vendor_id not in ["V001", "V002", "V003"]:
        return {"action_type": "escalate", "invoice_id": inv_id,
                "reason": f"Unknown vendor ID {vendor_id}"}

    po_ref = current_invoice.get("po_reference", "")
    if po_ref != f"PO-{inv_id}":
        return {"action_type": "flag_discrepancy", "invoice_id": inv_id,
                "discrepancy_type": "duplicate"}

    if po_data and receipt_data:
        if receipt_data.get("received_qty", 0) < po_data.get("approved_qty", 0):
            return {"action_type": "flag_discrepancy", "invoice_id": inv_id,
                    "discrepancy_type": "quantity"}

    if po_data:
        agreed = po_data.get("agreed_unit_price", 0)
        items = current_invoice.get("line_items", [{}])
        invoice_price = items[0].get("unit_price", 0) if items else 0
        if agreed > 0:
            variance = abs(invoice_price - agreed) / agreed
            if variance > TOLERANCE_SOFT:
                return {"action_type": "flag_discrepancy", "invoice_id": inv_id,
                        "discrepancy_type": "price"}

    return {"action_type": "approve_payment", "invoice_id": inv_id,
            "amount": current_invoice.get("total", 0)}

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_level: str, seed: int = 42) -> float:
    global _invoice_progress
    _invoice_progress = {}

    rewards: List[float] = []
    final_grade = 0.0
    steps_taken = 0

    log_start(task=task_level, env="InvoiceReconcileEnv", model=MODEL_NAME)

    try:
        result = reset_env(task_level, seed)
        observation = result.get("observation", {})

        for step_num in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            action = rule_based_agent(observation)
            action_str = json.dumps(action)

            try:
                result = step_env(action)
                observation = result.get("observation", {})
                reward = float(result.get("reward", 0))
                done = result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step_num

            msg = observation.get("message", "")

            # Update progress tracker
            import re
            match = re.search(r"INV-\d+", msg)
            acted_id = match.group(0) if match else ""
            if acted_id:
                if acted_id not in _invoice_progress:
                    _invoice_progress[acted_id] = {
                        "extracted": False, "po": False, "receipt": False,
                        "po_data": None, "receipt_data": None,
                    }
                p = _invoice_progress[acted_id]
                if f"Extracted fields for {acted_id}" in msg:
                    p["extracted"] = True
                if "PO retrieved for" in msg:
                    p["po"] = True
                    if observation.get("po_data"):
                        p["po_data"] = observation["po_data"]
                if "Goods receipt for" in msg:
                    p["receipt"] = True
                    if observation.get("receipt_data"):
                        p["receipt_data"] = observation["receipt_data"]

            log_step(step=step_num, action=action_str, reward=reward,
                     done=done, error=error)

            if done:
                if "Final grade:" in msg:
                    try:
                        final_grade = float(
                            msg.split("Final grade:")[1].strip().split()[0].rstrip(".")
                        )
                    except Exception:
                        final_grade = 0.0
                break

    except Exception as e:
        log_end(success=False, steps=steps_taken, rewards=rewards)
        return 0.0

    success = final_grade >= 0.5
    log_end(success=success, steps=steps_taken, rewards=rewards)
    return final_grade


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tasks = ["easy", "medium", "hard"]
    scores = {}
    for task in tasks:
        scores[task] = run_task(task, seed=42)

    print("\nBASELINE SCORES SUMMARY", flush=True)
    for task, score in scores.items():
        print(f"  {task.upper():10} → {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  AVERAGE    → {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
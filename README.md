# 🚚 Hybrid Quantum-Classical Supply Chain Optimization

![Status](https://img.shields.io/badge/Status-Research%20Framework-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Quantum](https://img.shields.io/badge/Quantum-Ready-purple)

> A **Two-Stage Stochastic Programming** framework that combines  
> **QAOA (Quantum Approximate Optimization)** for strategic decisions with  
> **MILP (Mixed-Integer Linear Programming)** for operational scheduling.

---

## ⚠️ Important: What This Project Actually Is

This is a **research/demonstration framework** - not a production-ready quantum advantage system.

| Claim | Reality |
|-------|---------|
| "Quantum-Enabled" | ⚠️ Quantum runs on simulator; real hardware via IBM (limited) |
| "Optimal" | ⚠️ Heuristic approximation with repair |
| "Production-Ready" | ⚠️ Research framework for learning/demonstration |

### Current Quantum Limitations (NISQ Era)

- **Simulator limit**: ~25 qubits (practical)
- **IBM Hardware limit**: 127 qubits (available backends)
- **Real-world networks**: Often exceed these limits → uses classical fallback

**This project is "Quantum-Ready"** - prepared for when hardware improves.

---

## 🎯 Problem Overview

We solve a **multi-echelon supply chain network design problem under demand uncertainty**:

- **Suppliers (S)**: Provide raw materials  
- **Warehouses (W)**: Store inventory and distribute products  
- **Customers (C)**: Retail tiers and factories with stochastic demand  

### Objective
Minimize **expected total cost**: Transportation + Inventory holding + Shortage penalties

---

## 🚀 The Hybrid Approach

### Stage 1 — Strategic Network Design
- Decide **which routes should exist** (binary decisions)
- Solved via **QUBO** using QAOA (quantum) or Simulated Annealing (classical)

### Stage 2 — Operational Planning
- Decide **how much to ship, store, and backorder**
- Solved using **MILP** for each demand scenario

---

## 🛠️ Installation

```bash
# Clone and install
git clone <repo>
cd supply-chain-optimization
uv sync

# Set up API keys (optional for real-world distances/quantum)
cp .env.example .env
# Edit .env with your keys
```

### Required API Keys (Optional)
- `ORS_API_KEY`: OpenRouteService for real road distances
- `IBM_QUANTUM_TOKEN`: IBM Quantum for real hardware execution

---

## 💻 Usage

### Run Optimization (CLI)
```bash
# Quick test with classical solver
uv run python -m supply_chain_optimization.main --test-mode --classical

# With QAOA simulator (small problems only)
uv run python -m supply_chain_optimization.main --test-mode

# With real IBM Quantum (requires token)
uv run python -m supply_chain_optimization.main --test-mode --use-ibm-quantum
```

### Web UI (Streamlit)
```bash
cd supply_chain_optimization/ui
uv run streamlit run app.py
```

---

## 📂 Project Structure

```
supply_chain_optimization/
├── hybrid/                 # Two-stage coordination
│   ├── coordinator.py      # Main solver
│   └── repair.py           # Feasibility repair
├── stage1_qubo/            # Strategic optimization
│   ├── qubo_builder.py     # QUBO construction
│   ├── qaoa_solver.py      # QAOA + IBM Quantum
│   └── constraints.py      # Penalty encoding
├── stage2_milp/            # Operational optimization
│   ├── milp_model.py       # PuLP MILP model
│   └── solver.py           # Multi-scenario solver
├── data/
│   ├── models.py           # Network entities
│   └── openrouteservice_client.py  # Real distances
├── ui/
│   └── app.py              # Streamlit dashboard
└── config.py               # Configuration
```

---

## ⚡ Known Limitations

### 1. Quantum Scalability
- QAOA on simulator: practical limit ~25 qubits
- Real problems often require classical fallback
- IBM Quantum: up to 127 qubits, but queue times can be hours

### 2. Stage 1/Stage 2 Disconnect
Stage 1 uses distance-based proxy costs; Stage 2 has complex dynamics (inventory, shortages). May not find globally optimal topology.

### 3. Repair Heuristic
QAOA produces noisy solutions that require post-processing repair. This is fundamental to NISQ devices.

### 4. Static Lead Times
Lead times are deterministic. Real supply chains have lead time variability.

---

## 📊 Benchmarks

| Config | QUBO Vars | Solver | Time |
|--------|-----------|--------|------|
| Tiny (3S, 2W, 5C) | ~30 | QAOA Sim | ~4s |
| Small (5S, 3W, 10C) | ~66 | Classical | ~15s |
| Medium (10S, 5W, 20C) | ~185 | Classical | ~2min |
| Large (20S, 10W, 50C) | ~770 | Classical | ~16min |

---

## 🎓 Learning Resources

This project demonstrates:
1. **QUBO formulation** for combinatorial optimization
2. **Qiskit QAOA** implementation
3. **Two-stage stochastic programming**
4. **Hybrid quantum-classical algorithms**

---

## 📝 License

MIT License - Use for learning, research, and experimentation.

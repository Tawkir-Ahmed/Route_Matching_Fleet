# app.py
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

# Optional: OpenAI chatbot
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False


# -----------------------------
# Utilities
# -----------------------------
def euclid(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def generate_dummy_network(n_nodes: int, n_routes: int, max_freq: int, seed: int):
    rng = np.random.default_rng(seed)

    nodes = pd.DataFrame({
        "node": list(range(n_nodes)),
        "x": rng.uniform(0, 100, size=n_nodes),
        "y": rng.uniform(0, 100, size=n_nodes),
    })

    # Create unique directed OD pairs (origin != destination)
    od_pairs = set()
    routes = []
    attempts = 0
    while len(routes) < n_routes and attempts < 5000:
        o = int(rng.integers(0, n_nodes))
        d = int(rng.integers(0, n_nodes))
        if o == d:
            attempts += 1
            continue
        if (o, d) in od_pairs:
            attempts += 1
            continue

        od_pairs.add((o, d))
        freq = int(rng.integers(1, max_freq + 1))
        routes.append({"route": len(routes), "origin": o, "dest": d, "freq": freq})
        attempts += 1

    routes = pd.DataFrame(routes)

    # Distance lookups
    coord = {int(r.node): (float(r.x), float(r.y)) for _, r in nodes.iterrows()}
    routes["loaded_dist"] = routes.apply(lambda r: euclid(coord[int(r.origin)], coord[int(r.dest)]), axis=1)
    routes["baseline_empty"] = routes.apply(lambda r: euclid(coord[int(r.dest)], coord[int(r.origin)]), axis=1)

    return nodes, routes, coord


def compute_pair_savings(routes: pd.DataFrame, coord: dict, epsilon: float):
    # savings_{rs} = (d(dr,or) + d(ds,os)) - (d(dr,os) + d(ds,or))
    pairs = []
    rlist = routes.to_dict("records")
    for i in range(len(rlist)):
        for j in range(i + 1, len(rlist)):
            r = rlist[i]
            s = rlist[j]
            or_, dr_ = int(r["origin"]), int(r["dest"])
            os_, ds_ = int(s["origin"]), int(s["dest"])

            base = euclid(coord[dr_], coord[or_]) + euclid(coord[ds_], coord[os_])
            paired = euclid(coord[dr_], coord[os_]) + euclid(coord[ds_], coord[or_])
            saving = base - paired

            pairs.append({
                "r": int(r["route"]),
                "s": int(s["route"]),
                "freq_r": int(r["freq"]),
                "freq_s": int(s["freq"]),
                "saving": float(saving),
                "is_valid": bool(saving >= epsilon),
            })

    pairs = pd.DataFrame(pairs)
    valid = pairs[pairs["is_valid"]].copy().reset_index(drop=True)
    return pairs, valid


def solve_route_matching(routes: pd.DataFrame, valid_pairs: pd.DataFrame):
    # CP-SAT integer program (ILP/MILP form):
    # y_rs integer 0..min(freq_r, freq_s)
    # sum_{pairs incident to r} y <= freq_r
    # maximize sum y * saving
    model = cp_model.CpModel()

    freq = {int(r.route): int(r.freq) for _, r in routes.iterrows()}

    y = {}  # (r,s) -> IntVar
    SCALE = 1000  # scale floats to ints for CP-SAT

    for _, row in valid_pairs.iterrows():
        r, s = int(row.r), int(row.s)
        ub = min(freq[r], freq[s])
        y[(r, s)] = model.NewIntVar(0, ub, f"y_{r}_{s}")

    # Frequency constraints per route
    for r in freq.keys():
        incident = []
        for (i, j), var in y.items():
            if i == r or j == r:
                incident.append(var)
        if incident:
            model.Add(sum(incident) <= freq[r])

    # Objective
    obj_terms = []
    for _, row in valid_pairs.iterrows():
        r, s = int(row.r), int(row.s)
        saving = float(row.saving)
        obj_terms.append(y[(r, s)] * int(round(saving * SCALE)))

    model.Maximize(sum(obj_terms) if obj_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0

    status = solver.Solve(model)

    result_rows = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for _, row in valid_pairs.iterrows():
            r, s = int(row.r), int(row.s)
            val = int(solver.Value(y[(r, s)]))
            if val > 0:
                result_rows.append({
                    "r": r,
                    "s": s,
                    "matched_trips": val,
                    "saving_per_trip": float(row.saving),
                    "total_saving": val * float(row.saving),
                })

    matches = pd.DataFrame(result_rows)
    if len(matches) == 0:
        return matches
    return matches.sort_values("total_saving", ascending=False).reset_index(drop=True)


def compute_totals(routes: pd.DataFrame, matches: pd.DataFrame):
    baseline_empty_total = float((routes["freq"] * routes["baseline_empty"]).sum())
    total_saving = float(matches["total_saving"].sum()) if len(matches) else 0.0
    post_empty_total = baseline_empty_total - total_saving
    pct = (total_saving / baseline_empty_total * 100.0) if baseline_empty_total > 0 else 0.0
    return baseline_empty_total, post_empty_total, total_saving, pct


def plot_network(nodes: pd.DataFrame, routes: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.scatter(nodes["x"], nodes["y"])
    for _, n in nodes.iterrows():
        ax.text(n["x"], n["y"], str(int(n["node"])), fontsize=9)

    node_xy = {int(r.node): (float(r.x), float(r.y)) for _, r in nodes.iterrows()}
    for _, r in routes.iterrows():
        o, d = int(r["origin"]), int(r["dest"])
        x1, y1 = node_xy[o]
        x2, y2 = node_xy[d]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"R{int(r['route'])}", fontsize=8)

    ax.set_title("Dummy network nodes + directed routes")
    ax.set_xlabel("x (miles)")
    ax.set_ylabel("y (miles)")
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# Chatbot helpers
# -----------------------------
def build_data_context(opt_state: dict, max_rows: int = 25) -> str:
    """Build a compact text context for the chatbot."""
    nodes = opt_state["nodes"]
    routes = opt_state["routes"]
    valid_pairs = opt_state["valid_pairs"]
    matches = opt_state["matches"]
    params = opt_state["params"]
    metrics = opt_state["metrics"]

    def df_block(name: str, df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return f"{name}: (empty)\n"
        return f"{name} (top {min(len(df), max_rows)} rows):\n{df.head(max_rows).to_csv(index=False)}\n"

    baseline_empty_total, post_empty_total, total_saving, pct = metrics

    model_desc = f"""
MODEL (route matching ILP solved with OR-Tools CP-SAT):
- Variable y[r,s] = integer number of matched trips between routes r and s (r<s).
- Objective: maximize sum_{r<s} y[r,s] * savings[r,s].
- Constraint: for each route r, sum_{s != r} y[r,s] <= freq[r].
- Valid pairs only: savings[r,s] >= epsilon.
Savings formula:
savings[r,s] = (dist(d_r, o_r) + dist(d_s, o_s)) - (dist(d_r, o_s) + dist(d_s, o_r))

Run parameters: n_nodes={params['n_nodes']}, n_routes={params['n_routes']}, max_freq={params['max_freq']},
epsilon={params['epsilon']}, seed={params['seed']}
Key metrics:
- baseline_empty_total={baseline_empty_total:.3f}
- post_empty_total={post_empty_total:.3f}
- total_saving={total_saving:.3f}
- savings_pct={pct:.3f}%
""".strip()

    context = (
        model_desc
        + "\n\n"
        + df_block("ROUTES", routes[["route", "origin", "dest", "freq", "loaded_dist", "baseline_empty"]])
        + df_block("VALID_PAIRS", valid_pairs[["r", "s", "saving", "freq_r", "freq_s"]].sort_values("saving", ascending=False))
        + df_block("MATCHES", matches if matches is not None else pd.DataFrame())
        + df_block("NODES", nodes)
    )

    # Keep context bounded
    if len(context) > 12000:
        context = context[:12000] + "\n\n[Context truncated to fit.]"
    return context


def get_openai_client(api_key: str | None):
    if not HAS_OPENAI:
        return None
    if api_key:
        return OpenAI(api_key=api_key)
    # If OPENAI_API_KEY is set in env / Streamlit secrets, this works:
    return OpenAI()


def ask_chatbot_openai(user_question: str, opt_state: dict, api_key: str | None, model_name: str):
    client = get_openai_client(api_key)
    if client is None:
        raise RuntimeError("OpenAI SDK not installed. Add `openai` to requirements and pip install openai.")

    system = (
        "You are a helpful assistant for a Streamlit demo of tactical truck route matching. "
        "Answer ONLY using the provided model description + tables. "
        "If the answer is not supported by the tables/context, say what is missing and how to get it. "
        "When giving numbers, cite the exact table column used (e.g., ROUTES.freq, MATCHES.total_saving)."
    )

    context = build_data_context(opt_state)

    # Keep a short chat history to reduce tokens
    history = st.session_state.get("chat_messages", [])
    history_tail = history[-8:]  # last few turns

    input_items = [{"role": "system", "content": system + "\n\nDATA CONTEXT:\n" + context}]
    input_items += history_tail
    input_items.append({"role": "user", "content": user_question})

    # Responses API supports list-of-messages input. :contentReference[oaicite:1]{index=1}
    resp = client.responses.create(
        model=model_name,
        input=input_items,
        store=False,  # optional: don't store
    )
    return resp.output_text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FLEET Route Matching (Dummy Optimization)", layout="wide")
st.title("FLEET Tactical Truck Route Matching (MILP Optimization)")

# Session state init
if "opt_state" not in st.session_state:
    st.session_state.opt_state = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.header("Dummy data settings")
    n_nodes = st.slider("Number of nodes", 6, 30, 12, 1)
    n_routes = st.slider("Number of routes", 4, 25, 10, 1)
    max_freq = st.slider("Max frequency per route", 1, 8, 3, 1)
    epsilon = st.slider("Min saving threshold ε (miles)", 0.0, 50.0, 5.0, 0.5)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.divider()
    st.subheader("Chatbot (optional)")
    enable_chatbot = st.checkbox("Enable chatbot", value=False)
    model_name = st.text_input("Model name", value="gpt-5")  # change if you want
    api_key_in = st.text_input("OPENAI_API_KEY (optional)", type="password", value="")
    st.caption("Tip: On Streamlit Cloud, set OPENAI_API_KEY in Secrets instead of typing here.")

    run = st.button("Run optimization", type="primary")

if run:
    nodes, routes, coord = generate_dummy_network(n_nodes, n_routes, max_freq, int(seed))
    all_pairs, valid_pairs = compute_pair_savings(routes, coord, float(epsilon))
    matches = solve_route_matching(routes, valid_pairs)
    baseline_empty_total, post_empty_total, total_saving, pct = compute_totals(routes, matches)

    # store for chatbot and for reruns
    st.session_state.opt_state = {
        "nodes": nodes,
        "routes": routes,
        "coord": coord,
        "all_pairs": all_pairs,
        "valid_pairs": valid_pairs,
        "matches": matches,
        "params": {
            "n_nodes": n_nodes,
            "n_routes": n_routes,
            "max_freq": max_freq,
            "epsilon": float(epsilon),
            "seed": int(seed),
        },
        "metrics": (baseline_empty_total, post_empty_total, total_saving, pct),
    }

# Show results if available
if st.session_state.opt_state is not None:
    opt = st.session_state.opt_state
    nodes = opt["nodes"]
    routes = opt["routes"]
    all_pairs = opt["all_pairs"]
    valid_pairs = opt["valid_pairs"]
    matches = opt["matches"]
    baseline_empty_total, post_empty_total, total_saving, pct = opt["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline empty miles", f"{baseline_empty_total:,.1f}")
    c2.metric("Post-matching empty miles", f"{post_empty_total:,.1f}")
    c3.metric("Total empty miles saved", f"{total_saving:,.1f}")
    c4.metric("Savings (%)", f"{pct:,.1f}%")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Nodes & Routes (dummy)")
        st.write("Nodes")
        st.dataframe(nodes, use_container_width=True)
        st.write("Routes")
        st.dataframe(routes, use_container_width=True)
        st.pyplot(plot_network(nodes, routes), clear_figure=True)

    with right:
        st.subheader("Valid pairs & matches")
        st.write(f"Valid pairs (saving ≥ ε): **{len(valid_pairs)}** out of **{len(all_pairs)}**")
        st.dataframe(
            valid_pairs.sort_values("saving", ascending=False)[["r", "s", "saving", "freq_r", "freq_s"]],
            use_container_width=True
        )

        if len(matches) == 0:
            st.info("No matches selected. Try lowering ε or increasing routes.")
        else:
            st.write("Selected matches")
            st.dataframe(matches, use_container_width=True)

            # Coverage check
            used = {int(r.route): 0 for _, r in routes.iterrows()}
            for _, m in matches.iterrows():
                used[int(m.r)] += int(m.matched_trips)
                used[int(m.s)] += int(m.matched_trips)

            coverage = pd.DataFrame({
                "route": list(used.keys()),
                "freq": [int(routes.loc[routes["route"] == r, "freq"].iloc[0]) for r in used.keys()],
                "matched": [used[r] for r in used.keys()],
                "unmatched": [int(routes.loc[routes["route"] == r, "freq"].iloc[0]) - used[r] for r in used.keys()],
            }).sort_values("route")

            st.write("Route coverage (matched vs available frequency)")
            st.dataframe(coverage, use_container_width=True)

    # -----------------------------
    # Chat section
    # -----------------------------
    st.divider()
    st.subheader("Chat about the data & optimization results")

    if not enable_chatbot:
        st.info("Enable the chatbot from the sidebar to ask questions about ROUTES / VALID_PAIRS / MATCHES / metrics.")
    else:
        if not HAS_OPENAI:
            st.error("`openai` package is not installed. Add it to requirements.txt and run: pip install openai")
        else:
            # Show chat history
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_q = st.chat_input("Ask anything about the model/data/results (e.g., 'Why pair (2,7) is valid?')")

            if user_q:
                st.session_state.chat_messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    try:
                        key_to_use = api_key_in.strip() if api_key_in.strip() else None
                        answer = ask_chatbot_openai(
                            user_question=user_q,
                            opt_state=opt,
                            api_key=key_to_use,
                            model_name=model_name.strip() or "gpt-5",
                        )
                        st.markdown(answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Chatbot error: {e}")

else:

    st.info("Set parameters in the sidebar and click **Run optimization**.")

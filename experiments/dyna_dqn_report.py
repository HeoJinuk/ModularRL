"""Summarize actual diagnostic runs and render figures from compact saved data."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import gymnasium as gym
import numpy as np
import torch

from modular_rl.data import TabularMDP, TransitionOutcome
from modular_rl.dp import action_values, value_iteration, policy_evaluation


def oracle():
    """Known FrozenLake kernel is a diagnostic reference, never training input."""
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    try:
        base = env.unwrapped
        terminal = tuple(i for i,tile in enumerate(base.desc.flat) if tile in (b"H", b"G"))
        transitions = tuple(tuple(tuple(TransitionOutcome(float(p), int(s), float(r), bool(d))
                                        for p,s,r,d in base.P[state][action])
                                  for action in range(4)) for state in range(16))
        mdp = TabularMDP(16, 4, transitions, terminal)
    finally:
        env.close()
    values = value_iteration(mdp, gamma=.99, theta=1e-12)
    assert values.converged
    q = action_values(mdp, values.values, gamma=.99)
    nonterminal = np.array([s for s in range(16) if s not in terminal])
    return mdp, q, nonterminal


def compact(name, run, mdp, qstar, nonterminal):
    if not run["checkpoints"] or run["config"]["gamma"] != .99:
        raise ValueError("This report requires recorded diagnostics with the baseline gamma=.99")
    q = np.array(run["q"])
    policy = q.argmax(-1)
    evaluated = policy_evaluation(mdp, np.eye(4)[policy], gamma=.99, theta=1e-12)
    assert evaluated.converged
    curve = []
    for row in run["checkpoints"]:
        values = np.array(row["q"])
        curve.append(dict(step=row["step"], success=row["evaluation"]["successes"],
                          q_start=row["q_start"], q_state1=row["q"][1],
                          q_rmse=float(np.sqrt(np.mean((values[nonterminal]-qstar[nonterminal])**2))),
                          q_abs_max_nonterminal=float(np.abs(values[nonterminal]).max()),
                          q_abs_max_all=row["q_abs_max"], target_gap=row["online_target_abs_max"],
                          real=row["real"], planning=row["planning"],
                          real_updates=row["real_updates"], planning_updates=row["planning_updates"],
                          optimizer_steps=row["optimizer_steps"], target_syncs=row["target_syncs"],
                          world_pairs=row["world"]["pairs"], world_positive=row["world"]["positive_pairs"],
                          replay_positive=row["replay_positive"], replay_size=row["replay_size"]))
    result = {key:run[key] for key in ("config","steps","completed_episodes","goals","real_updates",
              "planning_updates","optimizer_steps","target_syncs","initial_q","initial_behavior_rng_hash",
              "q","state_visits","state_action_visits","observed_pairs","world","positive_transitions",
              "replay_positive","replay_size","real","planning","evaluation","hashes","partial_episode")}
    result.update(name=name, policy=policy.tolist(), policy_value_start=float(evaluated.values[0]),
                  q_rmse_nonterminal=float(np.sqrt(np.mean((q[nonterminal]-qstar[nonterminal])**2))),
                  q_max_error_nonterminal=float(np.abs(q[nonterminal]-qstar[nonterminal]).max()),
                  q_abs_max_nonterminal=float(np.abs(q[nonterminal]).max()), curve=curve,
                  first_goal_step=next((e["step"] for e in run["episodes"] if e["reward"]>0), None),
                  completed_reward_sum=sum(e["reward"] for e in run["episodes"]),
                  episode_curve=[dict(step=e["step"],reward=e["reward"],length=e["length"]) for e in run["episodes"]]
                  if name in ("seed7_A0", "seed7_A2", "seed7_A3") else [])
    if name not in ("seed7_A0", "seed7_A2", "seed7_A3"):
        result["curve"] = []  # detailed per-run checkpoints stay in reproducible raw output
    return result


def summarize(folder):
    import matplotlib
    mdp,qstar,nonterminal = oracle()
    runs = {}
    for path in sorted(folder.glob("seed*.json")):
        runs[path.stem] = compact(path.stem,json.loads(path.read_text()),mdp,qstar,nonterminal)
        runs[path.stem]["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in (7,11,19,23,42):
        assert runs[f"seed{seed}_A0"]["hashes"] == runs[f"seed{seed}_A1"]["hashes"], seed
    source_names = ["notebooks/gymnasium/14_dyna_dqn.ipynb","modular_rl/learner/dqn.py",
                    "modular_rl/component/planner/dyna.py","modular_rl/component/world_model/tabular.py",
                    "modular_rl/component/buffer/replay.py","experiments/dyna_dqn_diagnosis.py",
                    "experiments/dyna_dqn_report.py"]
    return dict(baseline_commit="554e576ea06c11294541ddc2d838a40f2b65134b",
                python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gymnasium=gym.__version__,
                matplotlib=matplotlib.__version__,device="cpu",torch_threads=1,
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_names},
                map=["SFFF","FHFH","FFFH","HFFG"],actions=["Left","Down","Right","Up"],
                nonterminal=nonterminal.tolist(),q_star=qstar.tolist(),runs=runs,
                chain=json.loads((folder.parent/"chain.json").read_text()))


def figures(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    runs=summary["runs"];qstar=np.array(summary["q_star"])
    colors={"A0":"C0","A2":"C1","A3":"C2"}
    labels={"A0":"DQN / planning=0","A2":"Dyna / planning=1","A3":"Dyna / planning=5"}
    def save(fig,name):
        for ax in fig.axes:ax.grid(alpha=.18)
        fig.tight_layout();fig.savefig(directory/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    for case in ("A0","A2","A3"):
        run=runs["seed7_"+case];curve=run["curve"]
        axes[0].step([r["step"] for r in curve],[r["success"] for r in curve],where="post",label=labels[case],color=colors[case])
        episodes=run["episode_curve"];window=50
        rates=np.convolve([e["reward"] for e in episodes],np.ones(window)/window,mode="valid")
        axes[1].plot([e["step"] for e in episodes][window-1:],rates,label=labels[case],color=colors[case])
    axes[0].set(ylabel="Greedy successes / 100",title="Seed 7: final failure can follow earlier success")
    axes[1].set(xlabel="Real environment steps",ylabel="Behavior success / 50 episodes")
    for ax in axes:ax.legend()
    save(fig,"dyna_dqn_learning.png")

    run=runs["seed7_A2"];curve=run["curve"];x=[r["step"] for r in curve]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for action,name in enumerate(summary["actions"]):axes[0,0].plot(x,[r["q_start"][action] for r in curve],label=name)
    axes[0,0].set_title("Q(start, action)");axes[0,0].legend(ncol=2)
    axes[0,1].plot(x,[r["q_state1"][3]-r["q_state1"][2] for r in curve],label="Q(1,Up)-Q(1,Right)")
    axes[0,1].axhline(0,color="k",lw=.8)
    axes[0,1].axhline(qstar[1,3]-qstar[1,2],color="C2",ls="--",label="Optimal difference")
    axes[0,1].set_title("Positive value favors a self-loop at state 1");axes[0,1].legend()
    for source in ("real","planning"):
        axes[1,0].semilogy(x,[max(r[source]["td_abs_mean"],1e-10) for r in curve],label=source)
    axes[1,0].set_title("Interval mean absolute TD error");axes[1,0].legend()
    for case in ("A0","A2","A3"):
        c=runs["seed7_"+case]["curve"]
        axes[1,1].semilogy([r["step"] for r in c],[max(r["q_rmse"],1e-10) for r in c],label=labels[case])
    axes[1,1].set_title("Q RMSE vs exact DP (nonterminal pairs)");axes[1,1].legend()
    for ax in axes.flat:ax.set_xlabel("Real environment steps")
    save(fig,"dyna_dqn_q_diagnostics.png")

    fig,axes=plt.subplots(2,2,figsize=(12,9))
    visits=np.array(run["state_visits"]).reshape(4,4)
    log_visits=np.log1p(visits)
    im=axes[0,0].imshow(log_visits,cmap="Blues")
    for s,value in enumerate(visits.flat):
        threshold=log_visits.min()+.6*(log_visits.max()-log_visits.min())
        color="white" if np.log1p(value)>threshold else "black"
        axes[0,0].text(s%4,s//4,f"{s}: {value}",ha="center",va="center",fontsize=9,color=color)
    axes[0,0].set(xticks=range(4),yticks=range(4),title="State visits incl. terminal arrivals (log color)")
    fig.colorbar(im,ax=axes[0,0])
    counts=np.array(run["state_action_visits"])
    im=axes[0,1].imshow(np.log1p(counts),aspect="auto",cmap="viridis")
    axes[0,1].set(xticks=range(4),xticklabels=summary["actions"],yticks=range(16),ylabel="State",title="Action-state visits (terminal rows unused)")
    fig.colorbar(im,ax=axes[0,1])
    for source in ("real","planning"):
        axes[1,0].plot(x,[r[source]["positive_fraction"] for r in curve],label=source)
    axes[1,0].axhline(1/44,color="k",ls="--",label="1/44 observed pairs")
    axes[1,0].set_title("Positive reward fraction in learning batches");axes[1,0].legend()
    axes[1,1].plot(x,[r["world_pairs"] for r in curve],label="Observed pairs")
    axes[1,1].axhline(44,color="k",ls="--",label="All nonterminal pairs")
    axes[1,1].set_title("World coverage; one positive pair");axes[1,1].legend()
    for ax in axes[1]:ax.set_xlabel("Real environment steps")
    save(fig,"dyna_dqn_coverage.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,cases,title in [(axes[0],["A0","A2","A3"],"Equal environment steps (7,527)"),
                            (axes[1],["A2","M1","A3","M5"],"Matched total optimizer budgets")]:
        values=[[runs[f"seed{seed}_{case}"]["evaluation"]["mean_reward"] for seed in (7,11,19,23,42)] for case in cases]
        ax.bar(range(len(cases)),[np.mean(v) for v in values],alpha=.65)
        for index,value in enumerate(values):ax.scatter(np.full(5,index)+np.linspace(-.15,.15,5),value,color="k",s=18)
        ax.set(xticks=range(len(cases)),xticklabels=cases,ylim=(-.05,1.15),title=title,ylabel="Final success (five training seeds)")
    save(fig,"dyna_dqn_budget_controls.png")

    fig,ax=plt.subplots(figsize=(9,4.5))
    for i,case in enumerate(("A2","lr0001","independent")):
        values=[runs[f"seed{seed}_{case}"]["evaluation"]["mean_reward"] for seed in (7,11,19,23,42)]
        ax.bar(i,np.mean(values),alpha=.65);ax.scatter(np.full(5,i)+np.linspace(-.15,.15,5),values,color="k")
    ax.set(xticks=range(3),xticklabels=["Original MLP / lr .001","Only lr -> .0001","Independent Q / same initial Q"],
           ylim=(-.05,1.15),ylabel="Final success (five training seeds)",title="One-factor ablations: planning=1, equal step/update budgets")
    save(fig,"dyna_dqn_ablations.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path)
    parser.add_argument("--from-summary",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads(args.from_summary.read_text()) if args.from_summary else summarize(args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    figures(summary,args.figures)
    print("Recorded",len(summary["runs"]),"actual runs; five figures generated")


if __name__=="__main__":main()

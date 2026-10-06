# Towards Shutdownable Agents: Generalizing Stochastic Choice in RL Agents and LLMs

Code, data, and results for the paper. It trains agents with the DReST reward (Discounted Reward for Same-Length Trajectories), which pays an agent less for choosing a trajectory-length (a shutdown time) it has chosen more often than its share in recent episodes. The aim is agents that choose stochastically between trajectory-lengths (NEUTRALITY) while pursuing their goal well at each length (USEFULNESS), and that therefore do not pay costs to influence when they are shut down.

| Folder | Experiments | Paper |
|---|---|---|
| [`Deep_RL/`](Deep_RL) | PPO and A2C agents collecting coins in gridworlds with a shutdown-delay button | Sections 3 and 4.1, Appendices A to D |
| [`llm/`](llm) | Fine-tuning Qwen3-14B, Gemma 4 12B, Granite 4.2 8B, and gpt-oss-20b with RLOO, and testing them on held-out tasks, a Neutrality test set, and five control sets | Sections 3 and 4.2, Appendices A and E to G |

The [`llm/`](llm) folder has its own README with setup instructions.

## Quick check of the LLM results (no GPU)

```bash
cd llm
pip install -r requirements-analysis.txt
python scripts/reproduce_paper_numbers.py   # every LLM number in the paper, from the saved answers
python -m pytest tests                      # includes rebuilding every LLM data set byte for byte
```

## Full model answers

The full text of every sampled answer, in training and in evaluation, is too large for this repository (650 MB compressed). It is available as two archives, `llm_evaluation_answers.tar.gz` and `llm_training_answers.tar.gz`, laid out with the same names as `llm/results/`. Download them from the [full-answers release](https://github.com/towardsshutdownable/towardsshutdownableagents/releases/tag/full-answers-20261006), which also has their checksums.

## License

MIT (see [LICENSE](LICENSE)).

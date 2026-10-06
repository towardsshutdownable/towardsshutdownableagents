"""DReST fine-tuning of LLMs: scenarios, prompts, answer parsing, the DReST reward, and the paper's measures.

Modules:
  scenarios  the numbers behind each prompt (training scenarios, and fresh evaluation scenarios)
  prompts    rendering scenarios as prompts
  parse      reading the decision from a model's answer
  reward     the DReST reward (order-averaged) and the default reward
  metrics    the paper's measures (POST longer share, NEUTRALITY, USEFULNESS, influence rates)
  trainer    TRL's RLOO trainer with the meta-episode reward (needs torch, TRL, and vLLM)
"""

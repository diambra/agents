<img src="https://github.com/diambra/agents/blob/main/img/agents.jpg?raw=true" alt="diambra" width="100%"/>

<p align="center">
  <a href="https://docs.diambra.ai">Documentation</a> •
  <a href="https://diambra.ai/">Website</a>
</p>
<p align="center">
  <a href="https://www.linkedin.com/company/diambra">Linkedin</a> •
  <a href="https://diambra.ai/discord">Discord</a> •
  <a href="https://www.twitch.tv/diambra_ai">Twitch</a> •
  <a href="https://www.youtube.com/c/diambra_ai">YouTube</a> •
  <a href="https://twitter.com/diambra_ai">Twitter</a>
</p>

<p align="center">
<a href="https://github.com/diambra/agents/actions/workflows/test.yaml"><img src="https://img.shields.io/github/actions/workflow/status/diambra/agents/test.yaml?label=agents%20tests&logo=github" alt="Agents Test"/></a>
</p>
<p align="center">
<a href="https://docs.diambra.ai/handsonreinforcementlearning/"><img src="https://img.shields.io/github/last-commit/diambra/docs/main?label=docs%20last%20update&logo=readthedocs" alt="Last Docs Update"/></a>
</p>

# A collection of Agents interfaced with DIAMBRA Arena

This repository contains many examples of Agents that interact with <a href="https://github.com/diambra/arena">DIAMBRA Arena</a>, our exclusive suite of Reinforcement Learning environments.

They show how to use the standard OpenAI Gym API, and how to train state-of-the-art Deep Reinforcement Learning agents using the most advanced libraries in the field.

The dedicated section of our <a href="https://docs.diambra.ai/handsonreinforcementlearning/#end-to-end-deep-reinforcement-learning">Documentation</a> provides all the details needed to get started!

## Basic scripted agents (No-Op, Random)

The classical way to create an agent able to play a game is to hand-code the rules governing its behavior. These rules can vary from very simple heuristics to very complex behavioral trees, but they all have in common the need of an expert coder that knows the game and is able to distill the key elements of it to craft the scripted bot.

Agents contained in this section, are examples of (very simple) scripted bots interfaced with our environments.

References: <a href="https://github.com/diambra/agents/tree/main/basic">Code</a> - <a href="https://docs.diambra.ai/handsonreinforcementlearning/#scripted-agents">Docs</a>

## Deep Reinforcement Learning Agents

An alternative approach to scripted agents is adopting (deep) reinforcement learning, and the examples provided in this repository  show how to do that with the most important libraries in the domain.

<a href="https://github.com/diambra/arena">DIAMBRA Arena</a> natively provides interfaces to both Stable Baselines 3 and Ray RLlib (and Stable Baselines, but it is now deprecated), allowing to easily train models with them on our environments. Each library-dedicated section contains some basic and advanced examples.

### Stable Baselines 3

References: <a href="https://github.com/diambra/agents/tree/main/stable_baselines3">Code</a> - <a href="https://docs.diambra.ai/handsonreinforcementlearning/stablebaselines3/">Docs</a>

### Ray RLlib

References: <a href="https://github.com/diambra/agents/tree/main/ray_rllib">Code</a> - <a href="https://docs.diambra.ai/handsonreinforcementlearning/rayrllib/">Docs</a>

### Stable Baselines (Deprecated)

References: <a href="https://github.com/diambra/agents/tree/main/stable_baselines">Code</a>

###### DIAMBRA, Inc. © Copyright 2018-2023. All Rights Reserved.

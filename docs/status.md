---
layout: default
title:  Status
---

# {{ page.title }}

## Project Summary

The project focuses on developing an AI agent capable of autonomously navigating and mining in a simulated Minecraft environment. The agent must learn to identify valuable ores, select the appropriate pickaxe, and avoid hazards such as lava pits. The input to the system includes real-time environmental data (visual and positional), and the output consists of the agent’s actions, such as movement, mining, and decision-making. The goal of our agent is now to obtain a diamond in a real Minecraft world.
Initially, we explored using Craftium; however, due to compatibility issues with Rocky/RedHat Linux (used on HPC3), we transitioned to MineRL. The agent is currently being trained in `MineRLTreechop-v0`, which serves as a foundational environment before moving on to underground mining challenges.

## Approach

- **Environment Setup:** We successfully installed MineRL on HPC3 after resolving dependency conflicts with `pip` and `gradle`. Due to MineRL 1.0.2 being outdated and requiring manual installation (`python setup.py install`), this process involved extensive troubleshooting. 
- **Training Strategy:** We are planning to implement Proximal Policy Optimization (PPO) with convolutional neural network function approximators, utilizing on-policy learning to efficiently update the agent’s behavior. We also discovered a dataset of real player's input and gameplay data, so we will implement imitation learning to enhance the agent's learning.
- **Baseline Agent:** Currently, our agent takes random actions in different environments, ranging from simulating chopping tree to finding target block in small-room environments. This serves as a baseline for comparison when reinforcement learning is introduced for the main project.
- **Sequential Learning Progression:** The agent will first train on simpler environments, such as tree chopping and path finding, before transitioning to underground navigation and mining.
- **Implementation Challenges:** We encountered difficulties with recording long episode durations (~15 minutes), which impacted visualization and debugging. We are working on limiting episode length for more efficient data collection.

## Evaluation

We are evaluating the agent’s performance using both **quantitative** and **qualitative** methods:
### Quantitative Evaluation
The current goals are to track:
- The total number of valuable ores mined per episode.
- The agent’s survival time in the underground environment.
- The number of hazards (e.g., lava pits) successfully avoided.
- Comparison with the baseline random-action agent. We aim for at least a **10× improvement in ore collection** and a significant reduction in hazards encountered.

### Qualitative Analysis
- **Trajectory Maps:** Visualizing the agent’s movement patterns to understand decision-making.
- **Decision Trees for Tool Selection:** Ensuring the agent correctly picks tools based on ore types.
- **Heatmaps of Explored Areas:** Evaluating how effectively the agent navigates complex environments.


## Remaining Goals and Challenges

- **Reward Design:** Implementing a meaningful reward function to guide learning beyond random actions.
- **Imitation Learning** Allow agents to learn from real gameplays.
- **Training Refinements:** Adjusting hyperparameters (learning rate, batch size, discount factor) to optimize performance.
- **Underground Navigation:** Extending the agent’s ability from surface tasks to complex, procedurally generated underground environments.
- **HPC3 Compatibility Issues:** Ensuring smooth execution despite OS constraints.
- **Improved Recording Mechanisms:** Fixing recording issues to allow for better analysis of the agent’s actions.

## Resources Used

- **MineRL Documentation & GitHub:** Installation guidance and training setup.
- **Stack Overflow & AI/ML Forums:** Troubleshooting issues with dependencies (`pip`, `setuptools`, `gradle`).
- **Scientific Papers & Online RL Guides:** Designing the reinforcement learning framework.
- **GitHub Copilot & ChatGPT:** Assisting with debugging and coding, ensuring careful and responsible AI tool usage.
- **HPC3 Commands:** Running jobs efficiently (`srun --nodelist=hpc3-23-04 -A cs175_class --mem=10G -p standard --pty /bin/bash -i`).

## Video Summary 
[![AcastaCrafterProgressVideo](https://raw.githubusercontent.com/Y0507M/AcastaCrafter/main/docs/_assets/VideoCover.PNG)](https://www.youtube.com/)

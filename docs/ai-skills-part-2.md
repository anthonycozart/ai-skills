---
layout: post
title: "Intent and trust signals in AI instructions: Results"
date: 2026-04-05
display_date: "April 2026"
---

*This post shows empirical results for this idea ([part 1](../ai-skills-part-1/)) based on this methodology ([part 3](../ai-skills-part-3/)).*

## Takeaways

**1. Adoption of agent instructions is very low among non-enterprise users.** 

Just 3.5% of a 250k random sample of (project-level) repositories had markers of AI tooling, and only 1.1% had codified skill.md instructions. Those that did had a median of 4 instructions. Among these early adopters, instruction adoption is very similar across logarithmic-scale levels of activity.

![Number of instruction files per repository](../figures/1_1_skill_file_histogram.png)

**2. Instruction files span hundreds of topics in programming and data science.**

Many of the top 20 topics are about workflows, patterns, and execution steps. While they share this commonality, no topic dominates, and ~35% are unique and not easily clustered in groups with at least 15 free-form labels.

![Instruction share by object cluster](../figures/2_7_object_label_share.png)

**3. 3 in 5 instruction files reveal a “process specification” intent.**

The primary reason the human wrote the instruction was to tell the model to follow a specific sequence of steps, decision points, approval gates, or ordering requirements. 

![Instructions by primary intent](../figures/2_1b_intent_frequency.png)

**4. Some clusters, like design and front-end development, are an outsized share of intent categories.**

Humans are writing UI/UX instructions to align their preferences (duh).
	
![Object cluster by primary intent](../figures/2_2_object_intent_bubble.png)

**5. 95% of instructions give agents “adaptive” discretion.**

Very few instructions fully determine the process or output. Instead, most give agents frameworks to apply to situations. This makes sense… if the process is deterministic, then developers could write a script instead of an instruction.

![Primary intent by discretion](../figures/2_3_intent_discretion_matrix.png)

**6. Instructions with intents that have stronger trust explanation signals have more constraints.**

In Part 1, I explained constraints points toward a trust explanation, but their absence is ambiguous. In the chart below, we see that risk-mitigation and preference-alignment have an outsized share of 3rd and 4th quartile constraints per word. Put differently, compared to the “average” instruction, these instructions have more constraints and anticipate the agent’s failure.

![Constraint density by primary intent and discretion](../figures/2_6b_intent_discretion_constraint_quartiles.png)

**7. Instructions with process-specification intents suggest both trust and task explanations.**  

For prescribed discretion, shown on the left, the top quartile of constraints is most common. The developer added explicit prohibitions on top of a fixed sequence. That's excess specification beyond what task determinism requires, which leans toward trust explanation. BUT, (4th / 4th) is also common, suggesting some of these process-specification instructions may not be agent instructions, but instead boilerplate policy documents with many constraints and steps.

For adaptive discretion, shown on the right, the decision quartiles and constraint quartiles are similar. (4th / 4th) is the most common combination, showing above average constraints and above average decisions. Judgment is delegated but bounded.

![Decision vs constraint density for process-specification](../figures/2_8_process_spec_decision_constraint_matrix.png)

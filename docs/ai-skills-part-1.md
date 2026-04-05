---
layout: post
title: "Intent and trust signals in AI instructions: Idea"
date: 2026-04-05
display_date: "April 2026"
---

## Intent and trust signals in AI instructions

*This project analyzes AI instructions to understand developer intent and trust. The results are in [part 2](../ai-skills-part-2/), and the methodology in [part 3](../ai-skills-part-3/). Note that these posts are not on my website home page.*

## Intro

In 2025, tech and policy nerds started talking about agents. In 2026, tens of millions of people starting *using* them.

Agents are everywhere at tech companies, throughout tech stacks and in many, many conversations. Friends and peers are being told they have to use AI. They mostly love it. But they’re also being ranked, publicly, by their token usage. A few even feel like it’s being crammed down their throats. In my conversations, these friends are quite optimistic but say the expectation to delegate “everything” feels foolish, and they point to Amazon’s AI-related outages.[^1]

These perspectives are real but largely anecdotal. 

I’ve always had an empirical bent, and I’d love to see the data on broader usage at work. What kinds of developer tasks, for example, are agents handling? Claude Code is breathtaking when “it just works,” but what are the pros telling their agents? What are they concerned about? What frustrates them? What do they regularly fix?

To answer these questions, we’d need both the human-agent conversations and the backdrop: the code, artifacts, and other context that shapes agents’ behavior and humans’ responses. I obviously don’t have that data.

## My idea

A slightly different, narrower question may be more tractable: *What are developers and data scientists writing in repeatable instructions for their agents?* These instructions are increasingly common, and analyzing them could reveal what tasks are delegated repeatedly, and what the authors (human developers) are concerned about.

## What are agent instructions?

Both [OpenAI](https://developers.openai.com/codex/learn/best-practices) and [Anthropic](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) have detailed guides on when and why humans should use skill.md files to provide repeatable instructions. For example:

>“A skill is a set of instructions - packaged as a simple folder - that teaches Claude how to handle specific tasks or workflows. Skills are one of the most powerful ways to customize Claude for your specific needs. Instead of re-explaining your preferences, processes, and domain expertise in every conversation, skills let you teach Claude once and benefit every time.”

They continue with an analogy:

>“MCP provides the professional kitchen: access to tools, ingredients, and equipment. Skills provide the recipes: step-by-step instructions on how to create something valuable.”

## Why are we writing them?

I’ve been thinking about this question as a data science manager trying to decide how much time to divert from projects towards building the scaffolding to use AI more effectively.[^2] 

While the AI labs share [many reasons](https://www.anthropic.com/engineering/writing-tools-for-agents) to build this scaffolding, as a user, they boil down to two reasons:

1. **Task Explanation** - The user writes an instruction because the task genuinely has one right answer. This specificity reflects the nature of the task itself, not the user’s beliefs about the agent.  
2. **Trust Explanation** - The user writes an instruction because they don't trust the agent to “get it right,” so they spell it out. This specificity reflects their belief about the agent's limitations.

It’s helpful to think about how these explanations might reshape a distribution of delegated tasks and the value created by AI.

![Delegation value distribution](../figures/0_1_histogram.png)

The task explanation shifts the distribution to the right. Better context → better answers → in less time → more value. The trust explanation could shift the distribution to the right, or up, or both. Greater trust → fewer human-agent interactions → more value on individual tasks. Greater trust → more use cases → more value in aggregate.

The challenge, however, is the tasks and trust explanations are deeply intertwined. Instructions also reflect a task’s properties, like complexity and what’s at stake, as much as the developer’s trust in their agent. By providing instructions, humans get both better tasks and higher levels of trust. 

## Interlude…

Before going further, I want to acknowledge this has been an “occasional weeknight and weekends” project for me over the past month—during pockets of time during an otherwise very busy period for me and my spouse. I’ve worked on this idea when I’ve already been stretched by the day (managing a high performing team) and week (while we try to buy a house, plan our lives, pursue hobbies, etc). Brainstorming with Claude, and delegating most of the code writing, has helped make an already ambitious idea possible. We’ve taken thousands of turns together. I'm sure I've also made some silly choices and mistakes along the way.

## Empirical strategy

Here’s how we’ll turn this idea into an empirical exploration:

1. Let’s focus on skill.md files, which have a frontmatter, a body, and additional files. The frontmatter tells us the user’s problem: what it does and when to use it. It’s short, but has incredibly rich data.

2. We’ll sample the developer platform GitHub to create a corpus of instruction files.

3. For each file in our corpus, we’ll use LLMs to determine the:

| *Dimension* | *Description* | *Data* |
| :---- | :---- | :---- |
| **Object** | The aspect of agent behavior the instruction is shaping (e.g., API error handling) | Open-ended |
| **Functional intent** | The primary and secondary reasons a human wrote the instruction, that is, the gap or concern that motivated it | Categorical:<br>Context-provision<br>Preference-alignment<br>Risk-mitigation<br>Tool-orchestration<br>Process-specification |
| **Discretion** | The extent of decisions the agent must make to follow the instruction | Categorical:<br>Prescribed<br>Adaptive |
| **Decisions** | The distinct decision points to follow the instruction | Count |
| **Constraints** | The explicit constraints in the instruction (e.g., "always use X") | Count |

4. Finally, we’ll explore the task vs. trust explanations by examining the interactions between {functional intent, discretion, decisions, and constraints}. 

Here’s how I’m thinking about this. Intent is a “prior” on the likely explanation—think of this as our “first guess.”

* Risk-mitigation → Trust  
* Preference-alignment → Trust  
* Process-specification → Unclear  
* Context-provision → Task  
* Tool-orchestration → Task  
     
By adding a second dimension, Discretion, we can update our guess and possibly learn about the strength of the trust or task explanation signal.   

Lastly, we might be able to use decision and constraint counts, with caution. The presence of constraints points toward a trust explanation, but their absence is ambiguous. Low (or zero) constraint counts could reflect high baseline levels of trust, or that the task has only one right answer, or simply that the developer chose not to specify (for whatever reason).  
    
I had the idea to interact *intent* and *discretion*, but brainstorming with Claude really advanced my thinking here. It also made this nice graphic to summarize:

![Intent and discretion matrix](../figures/0_2_matrix.png)

*Want to see results? See [part 2](../ai-skills-part-2/).*

[^1]:  The outage [may not](https://www.aboutamazon.com/news/company-news/amazon-outage-ai-financial-times-correction) have been caused by 100% AI code; that it is being talked about this way, as a tangible concern even if apocryphal, is telling.

[^2]:  Agents.md files are similar. If a skill.md file is the recipe for a good summer salad, an agents.md is the preface telling you what you need to know [to make a killer salad](https://food52.com/story/24378-salt-fat-acid-heat-samin-nosrat-salad-tips) regardless of the season. There are also “soul” files, which define *“who”* an agent is rather than *how* to do something.
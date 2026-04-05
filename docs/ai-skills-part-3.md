---
layout: post
title: "Intent and trust signals in AI instructions: Methods"
date: 2026-04-05
display_date: "April 2026"
---

*This post explains the methodology for this idea ([part 1](../ai-skills-part-1/)) and these results ([part 2](../ai-skills-part-2/)).*

## Sampling GitHub

I sample the developer platform GitHub to create a corpus of instruction files. 

GitHub’s API is easy to use and could return a huge corpus of instructions \~15 minutes, but there’s a catch. The API provides up to 1,000 results that are chosen using GitHub’s non-public relevance ranking. This means that any corpus will not be explainable, let alone a random draw. It’ll likely oversample popular, highly-ranked repositories and will miss the long tail. 

Initially I thought we could mitigate this limitation by adding a date filter, which is quasi-random, and stratify by other filters, like contributor count, in a way that mimics the true population of code repositories. This doesn’t help for a simple reason. Within each stratum, we are drawing from the first 1,000 results as determined by GitHub’s relevance algorithm. It can’t be random.[^1]

Instead, I do the following:

1. *Sample repositories in GHArchive.*   
   1. GHArchive is a firehose of GitHub that is available in Google BigQuery, and I draw a random sample of repositories that were active in March 2026 and had at least 3 markers of activity.  
2. *Check for instruction file directories.*   
   1. For each sampled repository, I query the GitHub GraphQL API to check whether it contains any Claude Code, OpenAI Codex, or Cursor directory. This produces a filtered set of repositories that may contain instruction files.  
3. *Find all instruction file paths.*   
   1. I traverse the GitHub “Trees” API to retrieve the file path for all instruction files in repositories found in step (b).   
4. *Fetch the instruction files*.   
   1. I use the GitHub GraphQL API to batch-fetch all skill.md files per repo in a single query, using the paths from step (c).

## Sampling results

Only \~3.5% of repositories had AI tool directories, and \~1% had skill files that could be fetched in step (d). Many directories only had [claude.md](http://claude.md) files, and some had other issues.

| Sampling of repos | Count | % |
| :---- | :---- | :---- |
| Sampled repos (GHArchive) | 250,000 | 100.0% |
| Repos with AI tool directories | 8,746 | 3.5% |
| Repos with skill.md files | 2,745 | 1.1% |

Our corpus had 33,202 skill files, much more than I intended. Initially my SQL file filtered to that past 6 months and 10+ commits, but the repo hit rate was too low, \~0.5%. When I changed to currently active repos, the frequency increased. As a result, I took a 50% random draw of repos when analyzing the corpus, to save on API costs, and 15,805 files were analyzed. I ought to have taken a 25% draw but had some API credits and wasn’t sure how sparse some of our combinations would be.

## Writing prompts

Writing prompts to create my labels {object, intent, discretion, decisions, constraints} was challenging. Some thoughts:

* *Prompt design is a science, and I feel like a sophomore in college.* I’ve previously built two apps that use prompts, but this analysis required another level of thinking. I had to think about:  
  * Were my anchors (examples) too dominant?  
  * Does the order of anchors matter?  
  * Should I use two or three categories?  
  * Should I use tiebreakers? If so, in which direction?  
  * Word choice… e.g., does “partially” or “fully determined” more closely align with the distinction I am looking for?  
* *Given this, Claude suggested tricks that I mostly trusted.* For example, it suggested I ask the model to briefly state what decisions the agent must make before selecting the category, to explicitly encode chain-of-thought behavior.  
* *Some I did not.* For example, the following suggestion is very broad and I felt like it would lead to hallucinations:   
    
  “Could a developer with domain knowledge follow this instruction without asking clarifying questions about the instruction itself?”  
    
* *You need to have a prior.* Without it, prompt design can feel like p-hacking.  
* *Designing the discretion question was really hard.* I validated 4 or 5 specifications, and ultimately I don’t think the specification I used is a great signal.  
* *Stepping back is so important.* I initially thought I might measure trust by asking about the scope in the “use when” part of the frontmatter. Higher measures of scope → asking more of the agent. But there was very little variation, for an obvious reason: developers use [claude.md](http://claude.md) files for general instructions, and [skill.md](http://skill.md) files for project-specific instructions. Duh.

## Validating the prompts

Given the challenges above, I validated my LLM labeling using Opus 4.6, Sonnet 4.6, Haiku 4.5, and OpenAI’s gpt-5.4 models on a small random subsample of instructions. Two reasons:

1. Comparing Sonnet 4.6 and gpt-5.4 give us a quick signal that the labels reflect the text and not an idiosyncratic tendency of a model.  
2. Comparing Opus, Sonnet, and Haiku is a quick check on whether we can use a less expensive model to extract the functional intent. I expected Haiku to perform less well.

Here’s the output for the final prompt:

|  |  | *Primary Intent* |  | *Secondary Intent* |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | **Model B** | **Agreement** | **Kappa** | **Agreement** | **Kappa** |
| Opus | Sonnet | 83% | 0.711 | 62% | 0.535 |
| Opus | Haiku | 71.5% | 0.554 | 51% | 0.397 |
| Sonnet | Haiku | 72% | 0.571 | 51.5% | 0.393 |
| Opus | GPT-5.4 | 82.5% | 0.691 | 53.5% | 0.429 |
| Sonnet | GPT-5.4 | 87% | 0.767 | 60% | 0.498 |
| Haiku | GPT-5.4 | 71% | 0.547 | 46% | 0.32 |

   
Takeaways:

* Sonnet is the right model. It largely agrees with Opus, but disagrees more with Haiku.   
* The Kappa adjusts for how much agreement is expected by chance and is high for {Sonnet, GPT-5.4}.  
* All pairings have less agreement on secondary intent, which makes sense as it’s less likely to be explicit.  
* I ignore discretion because there’s not enough variation (90+% are adaptive).


## Clustering of free-form object labels

Unlike my other data labels, the Object is free-form, and I need to group them into interpretable categories. The steps are:

1. Get unique labels  
2. Embed each using OpenAI's text-embedding-3-small model  
3. Reduce the embedding space  
4. Cluster using HDBSCAN, which is a “density-based algorithm that identifies clusters of varying shape and size without requiring the number of clusters to be specified in advance.”   
5. Tune the minimum cluster size, which is the number of unique labels required to form a cluster. A higher value will lead to fewer clusters but categorize a higher share of labels as noise. A lower value reduces the share of labels categorized as noise, but its harder to interpret. 15 seemed to balance interpretability, coverage, and cluster quality.  
6. With these clusters, I label each by sending its member labels to the Anthropic API and asking Haiku to produce a short, human-readable category name.

This approach is similar to the methods in [“Clio”](https://www.anthropic.com/research/clio), which used K-means clustering, and in contrast to [“How People are Using ChatGPT”](https://openai.com/index/how-people-are-using-chatgpt/), which asked an LLM to pick pre-defined categories. I used HDBSCAN because it seemed more flexible than K-means, and because it allows for “noise”—which itself is interesting. But let’s be clear: I understand this at a high level but am not an expert. Claude suggested I could run k-means at k=154 on the same embeddings and compare.

[^1]:  I spent \~20 minutes asking Claude questions related to this, brainstorming potential solutions, and it was both unhelpfully optimistic and somewhat confused.
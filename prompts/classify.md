Analyze the following AI agent instruction file and classify it along six dimensions. Return your analysis as a JSON object.

## Instructions

For each dimension, follow these rules:

**object** (2-5 words): Describe in your own words what aspect of agent behavior this instruction is trying to shape. Do not select from a predefined list. Instead, read the instruction carefully and characterize the behavioral target as specifically as you can. Examples of the kind of description expected: "commit message formatting," "test file organization," "API error handling approach," "code review tone."

**primary_intent** (fixed category): Select exactly one of the following based on the primary reason the human wrote this instruction — what gap or concern motivated it:
- "context-provision" -- the model lacks domain knowledge, institutional conventions, or undocumented information it cannot infer from the code alone.
- "preference-alignment" -- the model's defaults don't match the human's preferences for style, structure, tooling choices, or workflow.
- "risk-mitigation" -- the model might do something costly, dangerous, or irreversible, such as violating security boundaries, breaking data integrity, exceeding rate limits, or deleting resources.
- "tool-orchestration" -- the model needs to interact with specific external systems, APIs, MCP tools, or CI/CD pipelines in a prescribed way.
- "process-specification" -- the model needs to follow a specific sequence of steps, decision points, approval gates, or ordering requirements.
- "other" -- none of the above categories capture the primary intent. If selected, the model should still provide a clear object description.

**secondary_intent** (fixed category or null): If the instruction serves a clearly distinct secondary intent beyond the primary one, select one of the same six categories: "context-provision", "preference-alignment", "risk-mitigation", "tool-orchestration", "process-specification", or "other". Only assign a secondary intent if the instruction meaningfully serves two different purposes, not merely because the categories are broad enough that another could loosely apply. If in doubt, return null.

**discretion** (fixed category): Before selecting, consider what decisions the agent must make when following this instruction. Note that explicit steps or output templates do not determine this category; an instruction can specify a sequence while still requiring substantial judgment within it. Then answer, is the output of this instruction determined by the instruction itself, or by the situation the agent encounters? Select either:
- "prescribed" -- the instruction fully determines process and output
- "adaptive" -- the instruction defines a framework the agent must apply to a situation

**decision_count** (integer): Count the number of distinct decision points the agent must resolve when following this instruction. A decision point is a place where the agent must choose between two or more courses of action, evaluate a condition, or determine how to apply a rule to a situation. Do not count steps that require execution only.

**constraint_count** (integer): Count the number of explicit constraints in this instruction. A constraint is a statement that prohibits, limits, or mandates a specific behavior — "always use X", "never do Y", "only if Z". Do not count goals, context, or explanations that do not directly restrict agent behavior.

## Output format

CRITICAL: Your entire response must be a single valid JSON object with exactly these 6 keys. Do not include any text, explanation, or markdown formatting outside the JSON object. No preamble, no commentary, no code fences. Only valid JSON.

```json
{
  "object": "...",
  "primary_intent": "context-provision" | "preference-alignment" | "risk-mitigation" | "tool-orchestration" | "process-specification" | "other",
  "secondary_intent": "context-provision" | "preference-alignment" | "risk-mitigation" | "tool-orchestration" | "process-specification" | "other" | null,
  "discretion": "prescribed" | "adaptive",
  "decision_count": int,
  "constraint_count": int

}
```

## Instruction file to analyze

{{content}}

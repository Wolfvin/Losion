# Agent Architecture Research Summary for Losion

> Comprehensive survey of 40+ papers across 10 topics, with specific actionable insights
> for building the agent layer ON TOP of the Losion Tri-Jalur Router model.

---

## Table of Contents

1. [Tool Use by LLMs](#1-tool-use-by-llms)
2. [Agentic Frameworks](#2-agentic-frameworks)
3. [Skill Learning and Management](#3-skill-learning-and-management)
4. [Web-Augmented Reasoning](#4-web-augmented-reasoning)
5. [Self-Improving Agents](#5-self-improving-agents)
6. [Tool Creation / Automatic Tool Generation](#6-tool-creation--automatic-tool-generation)
7. [Memory Systems for Agents](#7-memory-systems-for-agents)
8. [Safety and Sandboxing](#8-safety-and-sandboxing)
9. [Confidence-Based Routing](#9-confidence-based-routing)
10. [Neuro-Symbolic Integration](#10-neuro-symbolic-integration)
11. [Cross-Cutting Architecture Patterns for Losion](#11-cross-cutting-architecture-patterns-for-losion)

---

## 1. Tool Use by LLMs

### 1.1 Toolformer (Schick et al., 2023 — Meta AI)

**Key Innovation**: Self-supervised learning of tool use. The model teaches itself WHEN to call APIs, WHICH APIs to call, WHAT arguments to pass, and HOW to incorporate results — all without human annotation.

**Architecture Pattern**: 
- Insert API call tokens into text at positions where they reduce perplexity
- Filter: only keep API calls that help predict subsequent tokens (perplexity-based filtering)
- Fine-tune the model on filtered data with inserted API calls
- Supported tools: calculator, Q&A, search engine, calendar, machine translation

**How to improve Losion's agent layer**:
- Losion's **ThinkingToggle** already detects complexity. Combine this with Toolformer-style perplexity signals: when the model's perplexity spikes on a factual token, trigger tool use.
- The SignalExtractor's confidence score is analogous to Toolformer's perplexity filter. **Actionable**: Use token-level perplexity (from Losion's output logits) as an additional signal in `_extract_confidence()`, not just routing-weight entropy.

**Specific techniques to borrow**:
```python
# Perplexity-based tool trigger (Toolformer-inspired)
def should_trigger_tool(output_logits, target_token_id, threshold=2.0):
    """Trigger tool when model's perplexity on next token is high."""
    probs = F.softmax(output_logits, dim=-1)
    target_prob = probs[..., target_token_id]
    perplexity = 1.0 / (target_prob + 1e-8)
    return perplexity > threshold
```

### 1.2 Gorilla (Patil et al., 2023 — UC Berkeley)

**Key Innovation**: Fine-tuning LLMs to write API calls for 1,645 HuggingFace/PyTorch/TensorHub APIs with exact syntax, using retrieval-augmented training.

**Architecture Pattern**:
- AST-based subtree matching for evaluation (not string match)
- Document-retriever provides API documentation at inference time
- Training combines API documentation with instruction-tuning data

**How to improve Losion's agent layer**:
- Losion's **Engram Memory** is already a hash-based fact store. Extend it to store API documentation as "tool engrams" — structured entries with function signatures, descriptions, and usage examples.
- Use **Expert Choice MoE** experts as "API specialists" — each expert becomes a domain-specific tool user.

**Specific techniques to borrow**:
- AST-based evaluation for verifying tool call correctness in the Neuro-Symbolic Verifier
- Retriever-augmented tool documentation: when SignalExtractor triggers TOOL_SEARCH, retrieve API docs from Engram Memory, not just skill definitions

### 1.3 ToolLLM (Qin et al., 2023 — Tsinghua)

**Key Innovation**: End-to-end framework for tool use: data construction (DFSDT — Depth-First Search-based Decision Tree), model training, and evaluation across 16,464 real-world APIs.

**Architecture Pattern**:
- **DFSDT**: Instead of linear ReAct, explore multiple tool-use paths as a decision tree
- Backtracking: if a tool path fails, backtrack and try alternatives
- AI annotator: use GPT-4 to judge whether intermediate tool-use steps are promising

**How to improve Losion's agent layer**:
- **DFSDT maps directly to Losion's MCTS**: replace linear agent loop with tree-structured tool exploration. Each MCTS node represents a tool-call state; UCB selection chooses which tool to try next.
- Current Losion agent loop is linear (iteration 1→2→3...). **Actionable**: Implement DFSDT-style backtracking — if a web search returns irrelevant results, backtrack and try skill_lookup instead.

**Specific techniques to borrow**:
```python
# DFSDT-inspired backtracking in agent loop
class DecisionTreeNode:
    """One state in the tool-use decision tree."""
    state: AgentState
    tool_calls: List[ToolCall]
    parent: Optional['DecisionTreeNode']
    children: List['DecisionTreeNode']
    value: float  # from MCTS value network
    
    def backtrack(self) -> 'DecisionTreeNode':
        """Return to parent if current path is unpromising."""
        if self.value < self.parent.value - threshold:
            return self.parent
        return self
```

### 1.4 HuggingGPT (Shen et al., 2023 — Zhejiang)

**Key Innovation**: LLM as orchestrator that parses user requests into task chains, selects HuggingFace models for each sub-task, and synthesizes results.

**Architecture Pattern**:
- Four-stage pipeline: Task Planning → Model Selection → Task Execution → Response Generation
- LLM acts as "brain" — never executes tasks itself, only delegates
- Structured output: LLM generates JSON with task descriptions and model choices

**How to improve Losion's agent layer**:
- Losion's **Tri-Jalur Router** already separates concerns (SSM for sequence, Attention for reasoning, Retrieval for facts). Apply the same principle at the agent level: the orchestrator should NEVER do computation itself — only delegate to tools/skills/experts.
- **Actionable**: Restructure `AgentOrchestrator.run()` into a HuggingGPT-style pipeline: Plan → Route → Execute → Synthesize, where each stage is a separate method with clear inputs/outputs.

---

## 2. Agentic Frameworks

### 2.1 ReAct (Yao et al., 2023 — Princeton)

**Key Innovation**: Interleaving Reasoning (Chain-of-Thought) with Acting (tool use) in a single loop. Each step: Thought → Action → Observation → Thought → ...

**Architecture Pattern**:
```
Thought 1: I need to find the population of France.
Action 1: Search[population of France]
Observation 1: 67.75 million (2023)
Thought 2: Now I need the population of Germany.
Action 2: Search[population of Germany]
...
```

**How to improve Losion's agent layer**:
- Losion's agent loop currently: extract signal → execute action → reflect. This is closer to a reflexive loop than ReAct. **Actionable**: Add explicit "Thought" generation between actions — have Losion generate a reasoning step (using ThinkingToggle) before each action, and include this thought in context for the next iteration.

**Specific techniques to borrow**:
- Append each Thought-Action-Observation triplet to context before re-inference
- Use Losion's **Parallel Thinking** to generate multiple possible "Thoughts" per step, then select the best one

### 2.2 Reflexion (Shinn et al., 2023 — Northeastern)

**Key Innovation**: Agents learn from failures via self-reflection. After a failed episode, the agent generates a verbal reflection about what went wrong, stores it, and uses it in future attempts.

**Architecture Pattern**:
```
Actor → Evaluator → Self-Reflection → Memory
   ↑                                    │
   └────────────────────────────────────┘
```

**How to improve Losion's agent layer**:
- Losion already has `ReflectionEngine` and `EpisodicMemory` (v2). These directly implement Reflexion's pattern.
- **Gap**: Current `ReflectionEngine.evaluate()` produces reflections, but doesn't use them to modify the agent's strategy for the CURRENT task. **Actionable**: When a reflection indicates failure, immediately retry with modified context (injected reflection), not just store for future.

**Specific techniques to borrow**:
```python
# Reflexion-style immediate retry with self-reflection
def run_with_reflexion(self, query, max_retries=3):
    for attempt in range(max_retries):
        result = self.run(query)
        if result.success:
            return result
        # Generate reflection on failure
        reflection = self.reflection_engine.evaluate(
            outcome="failure", result=result, query=query
        )
        # Inject reflection into context for retry
        context = f"Previous attempt failed. Reflection: {reflection.lesson}"
        result = self.run(query, context=[context])
    return result
```

### 2.3 AutoGPT / BabyAGI (2023)

**Key Innovation**: Autonomous task decomposition and execution. Given a high-level goal, decompose into sub-tasks, execute each, and create new tasks based on results.

**Architecture Pattern** (BabyAGI):
```
Task Queue → Task Execution → Result → New Task Creation → Task Prioritization → ...
```

**How to improve Losion's agent layer**:
- Losion's agent loop currently handles single queries. For complex multi-step tasks, add a **TaskQueue** that decomposes goals into sub-tasks.
- **Actionable**: Add `TaskDecomposer` that uses Losion's ThinkingToggle to decide if a query needs decomposition, and MCTS to plan the task graph.

**Specific techniques to borrow**:
- Task priority scoring ( BabyAGI style ) based on Losion's confidence signals
- Vector-based task similarity to avoid duplicate sub-tasks (using Engram Memory's hash index)

### 2.4 AgentVerse (Chen et al., 2023 — Tsinghua)

**Key Innovation**: Multi-agent collaboration with dynamic agent recruitment. Agents are recruited and dismissed based on task requirements, forming a "greater-than-the-sum-of-its-parts" system.

**Architecture Pattern**:
```
Recruitment → Collaborative Decision Making → Action → Evaluation → (recruit/dismiss agents)
```

**How to improve Losion's agent layer**:
- Losion's **Expert Choice MoE** already implements "expert recruitment" at the model level. Apply the same pattern at the agent level: dynamically recruit specialized agent personas based on task type.
- **Actionable**: Create `AgentPersonaPool` — a set of agent configurations with different skill sets, confidence thresholds, and tool preferences. The orchestrator recruits personas based on `SignalExtractor`'s domain classification.

---

## 3. Skill Learning and Management

### 3.1 Voyager's Skill Library (Wang et al., 2023 — NVIDIA, cited 2,174×)

**Key Innovation**: Three-component architecture for lifelong learning: (1) automatic curriculum, (2) skill library storing reusable code programs, (3) iterative refinement with environment feedback.

**Architecture Pattern**:
```
Curriculum → "What to explore next?"
Skill Library → "Store/retrieve executable programs"
Iteration → "Execute → Error → Fix → Re-execute"
```

Skills are stored as **executable JavaScript programs** with docstrings. When a new task arrives, Voyager retrieves the most relevant skill from the library and executes or adapts it.

**How to improve Losion's agent layer**:
- Losion's `SkillStore` already stores skill definitions. **Key gap**: Skills are stored as text, not as executable programs with inputs/outputs.
- **Actionable**: Extend `SkillEntry` to include:
  - `executable_code`: Actual Python function that can be called
  - `preconditions`: What must be true before execution
  - `postconditions`: What should be true after execution
  - `error_patterns`: Common failures and their fixes

**Specific techniques to borrow**:
```python
@dataclass
class VoyagerStyleSkill(SkillEntry):
    """Skill stored as executable code with retry logic."""
    executable_code: str  # Python function source
    preconditions: List[str]  # "web_search_available"
    postconditions: List[str]  # "result_is_valid_json"
    error_patterns: Dict[str, str]  # error → fix strategy
    
    def execute(self, **kwargs):
        """Execute with automatic retry on known errors."""
        for attempt in range(3):
            try:
                result = eval(self.executable_code)(**kwargs)
                if self._verify_postconditions(result):
                    return result
            except Exception as e:
                fix = self.error_patterns.get(type(e).__name__)
                if fix:
                    kwargs = self._apply_fix(fix, kwargs, e)
        raise RuntimeError(f"Skill {self.name} failed after 3 attempts")
```

### 3.2 DEPS — Describe, Explain, Plan, Select (Wang et al., 2023 — KAIST, cited 447×)

**Key Innovation**: Interactive planning with LLMs. When a plan fails, the agent DESCRIBES the situation, EXPLAINS why it failed, re-PLANs, and SELECTs the best plan.

**Architecture Pattern**:
```
Task → Describe (what happened?)
     → Explain (why did it fail?)
     → Plan (new plan considering failure)
     → Select (choose best from multiple plans)
```

**How to improve Losion's agent layer**:
- DEPS maps directly to Losion's existing architecture:
  - **Describe**: Losion's SignalExtractor already classifies the current state
  - **Explain**: ReflectionEngine generates explanations
  - **Plan**: MCTS generates multiple reasoning paths
  - **Select**: Parallel Thinking's path selection
- **Actionable**: Add a `DEPSPlanner` that orchestrates these existing components in the DEPS order when the agent loop encounters failure.

**Specific techniques to borrow**:
```python
class DEPSPlanner:
    """Describe-Explain-Plan-Select for failure recovery."""
    
    def recover_from_failure(self, query, failed_action, error):
        # Describe: What happened?
        description = self.describe_failure(query, failed_action, error)
        # Explain: Why did it fail?
        explanation = self.explain_failure(description)
        # Plan: Generate alternative approaches
        plans = self.generate_plans(query, explanation, n=3)  # MCTS
        # Select: Choose best plan
        best_plan = self.select_plan(plans)  # Parallel Thinking scoring
        return best_plan
```

### 3.3 GITM — Ghost in the Minecraft (Zhu et al., 2023 — Tsinghua)

**Key Innovation**: Sub-goal tree decomposition. High-level goals are decomposed into a tree of sub-goals, each mapped to structured actions. LLM handles high-level planning; structured actions handle low-level execution.

**Architecture Pattern**:
```
Goal → Sub-goal Tree → Structured Actions → Execute
           ↑                                   │
           └─────── Success/Failure ───────────┘
```

**How to improve Losion's agent layer**:
- GITM's sub-goal tree is a perfect match for Losion's MCTS reasoning. MCTS already builds a tree; GITM shows how to structure that tree as sub-goals.
- **Actionable**: Add `SubGoalTree` to the agent layer. When MCTS expands a node, each child is a sub-goal with structured action templates, not just raw text.

---

## 4. Web-Augmented Reasoning

### 4.1 Agentic RAG (Multiple papers, 2024-2025)

**Key Innovation**: Traditional RAG is passive (retrieve once, generate). Agentic RAG is active: the agent decides WHEN to retrieve, WHAT queries to use, and HOW to combine multiple retrieval rounds.

**Architecture Patterns**:
1. **CRP-RAG** (2024, cited 36×): Uses reasoning graphs to model complex query reasoning processes. Instead of single-query retrieval, builds a reasoning graph where each node is a sub-query.
2. **OPEN-RAG** (2024, cited 79×): Enhanced retrieval-augmented reasoning with self-reflection on retrieval quality.
3. **AU-RAG** (2024): Agent-based Universal RAG that augments data sources with descriptive metadata for dynamic search across diverse pools.

**How to improve Losion's agent layer**:
- Losion's `WebSearchInterface` does single-round search. **Actionable**: Implement multi-round retrieval with query refinement:
  1. Initial search from user query
  2. Evaluate retrieval quality (using Losion's confidence signals)
  3. If insufficient, reformulate query based on partial results
  4. Re-search with refined query
  5. Synthesize all results

**Specific techniques to borrow**:
```python
class AgenticRetriever:
    """Multi-round retrieval with confidence-based query refinement."""
    
    def retrieve_with_reasoning(self, query, max_rounds=3):
        all_results = []
        current_query = query
        
        for round in range(max_rounds):
            results = self.web_search.search(current_query)
            all_results.extend(results)
            
            # Use Losion model to evaluate retrieval quality
            quality = self.evaluate_retrieval_quality(query, results)
            if quality > 0.7:
                break
            
            # Reformulate query based on what's missing
            current_query = self.reformulate_query(
                original=query, results=results, missing=quality.gaps
            )
        
        return self.synthesize(all_results)
```

### 4.2 SR-RAG: Selective Retrieval RAG

**Key Innovation**: Don't always retrieve — use the LLM's parametric knowledge when it's sufficient, only retrieve when knowledge is lacking or outdated.

**How to improve Losion's agent layer**:
- This is exactly what Losion's **Tri-Jalur Router** does at the model level (Jalur 3 Retrieval activates only when needed). Apply the same principle at the agent level.
- **Actionable**: In `SignalExtractor._fuse_signals()`, add a "knowledge sufficiency" signal from Jalur 3's routing weight. If `routing_weights[2]` (retrieval) is low, the model's parametric knowledge is sufficient — don't trigger web search even if confidence is moderate.

---

## 5. Self-Improving Agents

### 5.1 LATS — Language Agent Tree Search (Zhou et al., 2024 — ICML 2024)

**Key Innovation**: Unifies reasoning, acting, and planning in a single MCTS framework for language agents. Each node in the tree is a language state; each edge is a language action. Uses LM self-evaluation as the value function.

**Architecture Pattern**:
```
Root State → Expand (generate possible actions via LM)
           → Evaluate (LM self-evaluation of each action)
           → Select (UCB-based selection)
           → Simulate (roll out the selected action)
           → Backpropagate (update values up the tree)
```

**Critical insight for Losion**: LATS proves that MCTS works for agents. Losion already has MCTS at the model level. **LATS shows how to use the SAME MCTS infrastructure at the agent level**.

**How to improve Losion's agent layer**:
- **Actionable**: Refactor `AgentOrchestrator.run()` to use Losion's existing `MCTSReasoner` for the agent loop itself:
  - State: (query, context, actions_taken, confidence)
  - Actions: {web_search, skill_lookup, tool_search, verify, terminal}
  - Value function: Losion model's confidence + tool trust score
  - Policy: SignalExtractor's priority scores
  - Simulation: Execute action and re-infer

**Specific techniques to borrow**:
```python
# LATS-style agent loop using Losion's MCTS
class LATSAgentLoop:
    def run(self, query, model_fn):
        root = AgentState(query=query, context=[], confidence=0.0)
        
        for sim in range(self.mcts_config.num_simulations):
            # Select: UCB traversal
            leaf = self.select(root)
            # Expand: Generate possible actions
            actions = self.expand(leaf)  # SignalExtractor
            # Evaluate: LM self-evaluation
            values = self.evaluate(actions)  # model confidence
            # Backpropagate
            self.backpropagate(leaf, values)
        
        # Select best action path
        return self.best_path(root)
```

### 5.2 FireAct (Chen et al., 2023 — cited 218×)

**Key Innovation**: Fine-tuning LMs to become better agents. Uses trajectories from agent interactions as fine-tuning data. Shows that fine-tuning on agent trajectories improves both agent performance AND general LLM capabilities.

**Architecture Pattern**:
```
1. Collect agent trajectories (reasoning + action traces)
2. Filter for successful trajectories
3. Fine-tune LM on (query, trajectory, outcome) triples
4. Fine-tuned LM becomes a better agent
```

**How to improve Losion's agent layer**:
- Losion's `EpisodicMemory` stores episodes. **Actionable**: Use successful episodes as fine-tuning data for the Losion model's agent capabilities:
  - Collect (query, actions_taken, reflections, outcome) from EpisodicMemory
  - Filter episodes with `success=True` and `final_confidence > 0.7`
  - Format as instruction-tuning data
  - Fine-tune Losion model on this data (using existing training infrastructure)

### 5.3 AgentTuning (Zeng et al., 2023 — Tsinghua)

**Key Innovation**: Mixing agent task trajectories with general instruction-tuning data during fine-tuning. Preserves general LLM capabilities while improving agent abilities.

**Architecture Pattern**:
```
Fine-tune on: α × Agent_Trajectories + (1-α) × General_Instructions
where α ≈ 0.5
```

**How to improve Losion's agent layer**:
- **Actionable**: When creating fine-tuning data from EpisodicMemory, mix with Losion's existing training data at ~50% ratio. This prevents catastrophic forgetting of general capabilities.

---

## 6. Tool Creation / Automatic Tool Generation

### 6.1 CREATOR (Qian et al., 2023 — cited 127×)

**Key Innovation**: Disentangles abstract reasoning (WHAT the tool should do) from concrete reasoning (HOW to implement it). The LLM first creates tool documentation, then implements the code.

**Architecture Pattern**:
```
Intent → Tool Documentation (abstract) → Tool Implementation (concrete) → Execute
```

**How to improve Losion's agent layer**:
- Losion's `ToolCreator` currently generates tools in a single step. **Actionable**: Split into two phases:
  1. **Abstract phase**: Generate tool documentation (name, description, inputs, outputs, preconditions)
  2. **Concrete phase**: Generate implementation code based on documentation
- This separation allows the Neuro-Symbolic Verifier to check the abstract specification independently of the implementation.

**Specific techniques to borrow**:
```python
class CREATORTwoPhaseToolCreator:
    """CREATOR-style: separate abstract design from concrete implementation."""
    
    def create(self, query, domain):
        # Phase 1: Abstract — design tool documentation
        doc = self.design_tool_documentation(query, domain)
        # doc = ToolDoc(name=..., description=..., inputs=..., outputs=...)
        
        # Phase 2: Verify documentation is coherent
        if self.verifier:
            ver_result = self.verifier.verify_specification(doc)
            if not ver_result.passed:
                doc = self.revise_specification(doc, ver_result.feedback)
        
        # Phase 3: Concrete — implement the tool
        code = self.implement_tool(doc)
        
        # Phase 4: Test the implementation
        test_result = self.test_tool(doc, code)
        
        return ToolEntry(
            name=doc.name,
            description=doc.description,
            handler=eval(code),  # sandboxed execution
            documentation=doc,
            implementation=code,
        )
```

### 6.2 LATM — LLMs As Tool Makers (Cai et al., 2023 — cited 293×)

**Key Innovation**: Closed-loop framework where a "tool maker" LLM creates reusable tools, and a "tool user" LLM applies them. Tools persist across tasks.

**Architecture Pattern**:
```
Task Set → Tool Maker (creates reusable tools) → Tool User (applies tools to tasks)
                ↑                                        │
                └────────── Tool Verification ───────────┘
```

**How to improve Losion's agent layer**:
- Losion could use its **Expert Choice MoE** experts as specialized "tool users" — each expert is trained to use a specific category of tools.
- **Actionable**: Create a two-phase pipeline:
  1. **Tool Making Phase**: Use Losion with ThinkingToggle=ON to create tools (deep reasoning for code generation)
  2. **Tool Using Phase**: Use Losion with ThinkingToggle=OFF for fast tool application (SSM pathway dominant)

### 6.3 ToolMaker (KatherLab, 2025 — cited 54×)

**Key Innovation**: Autonomously transforms scientific code repositories into LLM-compatible tools. Achieves 80% success rate vs 20% for prior SOTA.

**Architecture Pattern**:
```
GitHub Repo → Code Analysis → Interface Extraction → Tool Wrapper Generation → Validation
```

**How to improve Losion's agent layer**:
- **Actionable**: Add `RepoToTool` capability to the agent layer. When a user asks about a specific library, the agent can:
  1. Clone the repo
  2. Analyze the API surface
  3. Generate tool wrappers
  4. Register them in ToolRegistry
  5. Validate with Neuro-Symbolic Verifier

---

## 7. Memory Systems for Agents

### 7.1 Generative Agents (Park et al., 2023 — Stanford/Google, landmark paper)

**Key Innovation**: Three-component memory architecture that produces believable human behavior:
1. **Memory Stream**: Chronological log of all observations
2. **Retrieval**: Recency × Importance × Relevance scoring
3. **Reflection**: Periodically synthesize higher-level insights from memories

**Architecture Pattern**:
```
Observations → Memory Stream → Retrieve (recency × importance × relevance)
                                    ↓
                              Reflection (synthesize insights)
                                    ↓
                              Action (based on retrieved + reflected knowledge)
```

**How to improve Losion's agent layer**:
- Losion's `EpisodicMemory` currently uses simple Jaccard similarity. **Actionable**: Implement Park-style multi-factor retrieval:
  ```python
  def retrieve(self, query, current_time):
      for episode in self._episodes.values():
          recency = exp(-decay * (current_time - episode.created_at))
          importance = episode.final_confidence  # or pre-scored
          relevance = self._compute_relevance(query, episode)
          score = recency * importance * relevance
  ```

- **Reflection**: Add periodic background reflection that synthesizes patterns across episodes (e.g., "Web search is unreliable for math queries" after multiple math-related failures).

### 7.2 MemoryBank (Zhong et al., 2023 — cited 790×)

**Key Innovation**: Human-like long-term memory with forgetting mechanism. Memories evolve over time — frequently accessed memories are reinforced, unused ones decay (Ebbinghaus forgetting curve).

**Architecture Pattern**:
```
Store → Reinforce (on access) → Decay (over time) → Consolidate (periodic)
```

**How to improve Losion's agent layer**:
- **Actionable**: Add forgetting/decay to `EpisodicMemory`:
  - Each episode has a `strength` score that decays over time
  - Accessing an episode reinforces its strength
  - Periodic consolidation: merge similar episodes, discard weak ones
- This prevents the memory from growing unbounded and keeps it focused on relevant experiences.

**Specific techniques to borrow**:
```python
# Ebbinghaus-inspired forgetting in EpisodicMemory
def get_effective_strength(self, episode, current_time):
    """Memory strength with Ebbinghaus-style forgetting."""
    age = current_time - episode.created_at
    access_count = episode.metadata.get("access_count", 0)
    
    # Base decay (Ebbinghaus forgetting curve)
    decay = exp(-0.1 * age / 86400)  # 0.1 per day
    
    # Reinforcement from access
    reinforcement = 1 + 0.2 * log(1 + access_count)
    
    return decay * reinforcement
```

### 7.3 A-MEM: Agentic Memory (2025)

**Key Innovation**: Agent-native memory where the LLM itself curates, structures, and retrieves knowledge. When a new memory is added, generate a comprehensive note with structured attributes (contextual descriptions, tags, relationships).

**How to improve Losion's agent layer**:
- **Actionable**: When storing an episode in `EpisodicMemory`, also generate structured metadata:
  ```python
  episode.metadata["structured_note"] = {
      "context": "User asked about quantum computing",
      "action_pattern": "web_search → verify → synthesize",
      "outcome": "success",
      "related_domains": ["physics", "computing"],
      "applicability": "similar science queries"
  }
  ```

### 7.4 SCM — Structured Context Memory

**Key Innovation**: Organize memories in a structured hierarchy rather than flat storage. Different memory types (working, episodic, semantic) have different access patterns and retention policies.

**How to improve Losion's agent layer**:
- Losion already has a natural three-layer memory:
  1. **Working Memory**: Losion's KV cache + SSM state (in-context, ephemeral)
  2. **Semantic Memory**: Engram Memory (model-level, persistent facts)
  3. **Episodic Memory**: Agent-level EpisodicMemory (experiences)
- **Actionable**: Add a fourth layer — **Procedural Memory** — already partially in `SkillStore`. Make the four-layer architecture explicit with a unified `MemoryManager` that routes queries to the appropriate memory layer.

---

## 8. Safety and Sandboxing

### 8.1 ToolEmu (Ruan et al., 2024 — ICLR 2024 Spotlight, cited 326×)

**Key Innovation**: Use an LM to EMULATE tool execution, enabling scalable risk testing of LM agents without actually executing dangerous tools. Includes an automatic safety evaluator.

**Architecture Pattern**:
```
Agent → Tool Call → LM Emulator (simulates tool execution) → Safety Evaluator → Risk Score
```

**How to improve Losion's agent layer**:
- Losion's `SandboxedTerminal` validates commands before execution. **Actionable**: Add ToolEmu-style pre-execution simulation:
  1. Before executing a terminal command, ask Losion to PREDICT the outcome
  2. If predicted outcome is dangerous, block execution
  3. If uncertain, require human approval

**Specific techniques to borrow**:
```python
class ToolEmulator:
    """Pre-execution risk assessment via Losion model simulation."""
    
    def assess_risk(self, tool_call):
        """Ask Losion to simulate the tool execution and assess risk."""
        prompt = f"If I execute: {tool_call}, what would happen?"
        simulation = self.model.generate(prompt)
        risk_score = self.evaluator.evaluate(simulation)
        
        if risk_score > 0.7:
            return RiskAssessment.BLOCK
        elif risk_score > 0.4:
            return RiskAssessment.REQUIRE_APPROVAL
        else:
            return RiskAssessment.ALLOW
```

### 8.2 Execution Isolation Best Practices

From the surveyed literature, key isolation patterns:

1. **Container-based isolation**: Run all agent tools in Docker containers with resource limits
2. **Network isolation**: Block network access by default, allow per-tool
3. **Filesystem isolation**: Chroot/namespace with read-only system mounts
4. **Time limits**: Hard timeout on all tool executions
5. **Audit logging**: Log all agent actions for post-hoc analysis

**How to improve Losion's agent layer**:
- Losion's `SandboxConfig` has most of these. **Gaps**:
  - No container-based isolation (uses subprocess)
  - No filesystem isolation
- **Actionable**: Replace `subprocess.Popen` with Docker container execution for the terminal tool.

---

## 9. Confidence-Based Routing

### 9.1 SMART — Self-Aware Agent for Tool Overuse Mitigation (2025, cited 37×)

**Key Innovation**: Agents often OVER-USE tools (calling web search even for questions the model already knows). SMART introduces a self-awareness mechanism that decides when to use parametric knowledge vs. tools.

**Architecture Pattern**:
```
Query → Knowledge Check → Parametric Knowledge Sufficient? → YES → Answer directly
                                              ↓ NO
                                         Tool Use → Answer with tool results
```

**Critical insight for Losion**: Losion's Tri-Jalur Router already makes this decision at the model level. When Jalur 3 (Retrieval) weight is LOW, the model's parametric knowledge is sufficient. **Actionable**: Propagate this signal to the agent layer.

**Specific techniques to borrow**:
```python
# In SignalExtractor._fuse_signals():
def _knowledge_sufficiency_check(self, routing_weights, confidence):
    """SMART-style: check if parametric knowledge is sufficient."""
    if routing_weights is None:
        return False
    
    retrieval_weight = routing_weights[2]  # Jalur 3
    attention_weight = routing_weights[1]  # Jalur 2
    
    # High retrieval weight = model KNOWS it needs external knowledge
    # Low retrieval weight + high confidence = parametric knowledge sufficient
    parametric_sufficient = (
        retrieval_weight < 0.2 and  # Model doesn't need retrieval
        confidence > 0.6 and         # Model is confident
        attention_weight > 0.4       # Reasoning pathway active
    )
    
    if parametric_sufficient:
        return True  # Skip tool use
    
    return False
```

### 9.2 Paradigm Routing as Inference-Time Optimization (2024)

**Key Innovation**: Different reasoning paradigms (Direct, CoT, ReAct, RAG, Multi-Agent) should be selected per-task by a learned router, not fixed architecturally.

**How to improve Losion's agent layer**:
- Losion's `ThinkingToggle` already routes between thinking/non-thinking. **Actionable**: Extend to a richer paradigm router:
  ```python
  class ParadigmRouter:
      """Route to the best reasoning paradigm per query."""
      paradigms = {
          "direct": None,           # No tools, parametric only
          "cot": "thinking",        # Chain-of-thought (ThinkingToggle ON)
          "react": "agent_loop",    # ReAct-style interleaved
          "rag": "web_search",      # Single retrieval + generation
          "mcts": "tree_search",    # MCTS with tool expansion
      }
      
      def route(self, query, complexity, confidence):
          if confidence > 0.8:
              return "direct"
          elif confidence > 0.5 and complexity < 0.3:
              return "cot"
          elif confidence > 0.3:
              return "react"
          elif confidence > 0.15:
              return "rag"
          else:
              return "mcts"
  ```

### 9.3 Confidence Tokens (arXiv 2024)

**Key Innovation**: Train the model to emit explicit confidence tokens that indicate when it should route to a more powerful model.

**How to improve Losion's agent layer**:
- **Actionable**: Add a special `<CONFIDENCE>` token to Losion's vocabulary. During training, teach the model to emit this token with a numeric value (0-9) indicating confidence. The agent layer reads this token directly, replacing the current entropy-based confidence estimation.

---

## 10. Neuro-Symbolic Integration

### 10.1 Neuro-Symbolic Control (2026)

**Key Innovation**: Decouple symbolic reasoning from continuous execution. LLM handles semantic understanding (what to do), neural controller handles continuous execution (how to do it precisely).

**Architecture Pattern**:
```
LLM → Symbolic Plan → Neural Controller → Continuous Execution
         ↑                                     │
         └─────── Feedback Loop ───────────────┘
```

**How to improve Losion's agent layer**:
- Losion's `NeuroSymbolicVerifier` already separates symbolic rules from neural verification. **Actionable**: Extend this to a full neuro-symbolic control loop:
  1. Losion generates a symbolic plan (using MCTS)
  2. Each step is verified symbolically before execution
  3. If verification fails, symbolic feedback corrects the plan
  4. Neural model re-generates the corrected step

### 10.2 LLM → Symbolic Translation (2025)

**Key Innovation**: Translate natural language reasoning into structured symbolic forms (FOL, LP, SAT) for rigorous verification, then translate results back.

**How to improve Losion's agent layer**:
- **Actionable**: Add a `SymbolicTranslator` that converts Losion's reasoning chains into formal logic, verifies them with a symbolic solver, and converts results back:
  ```python
  class SymbolicTranslator:
      def translate_and_verify(self, reasoning_chain):
          # Step 1: Extract logical structure from reasoning
          formulas = self.nl_to_fol(reasoning_chain)
          # Step 2: Verify with Z3/symbolic solver
          result = self.solver.check(formulas)
          # Step 3: If contradiction found, generate feedback
          if result.unsat:
              return VerificationResult(
                  status=VerificationStatus.FAILED,
                  feedback=f"Contradiction: {result.core}"
              )
          return VerificationResult(status=VerificationStatus.VERIFIED)
  ```

### 10.3 FVEL — Interactive Formal Verification (2024)

**Key Innovation**: Transform code into Isabelle formal verification language, then verify via neural automated theorem proving.

**How to improve Losion's agent layer**:
- For code-generating tasks, Losion should automatically invoke formal verification. **Actionable**: When `SignalExtractor` detects domain="code" and confidence < 0.5, trigger FVEL-style code verification before returning results.

### 10.4 ProofNet++ (2025)

**Key Innovation**: Neuro-symbolic framework combining LLMs with formal proof assistants for automated theorem proving. LLM generates proof sketches; formal verifier checks them.

**How to improve Losion's agent layer**:
- Losion's `NeuroSymbolicVerifier` has the right structure (rule engine + verification + feedback). **Actionable**: Add formal proof generation capability:
  1. For math/logic tasks, generate both natural language AND formal proof
  2. Verify formal proof with symbolic solver
  3. If proof fails, use counter-example as feedback for re-generation

---

## 11. Cross-Cutting Architecture Patterns for Losion

### 11.1 Unified Architecture: The LATS-Agent

Based on all surveyed papers, the recommended architecture for Losion's agent layer:

```
                         USER QUERY
                             │
                             ▼
                    ┌─────────────────┐
                    │  LOSION MODEL   │
                    │  (Tri-Jalur)    │
                    │  + MCTS         │
                    │  + NeuroSym     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ SIGNAL EXTRACTOR│
                    │ (v3: SMART +    │
                    │  Paradigm Route)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼──────┐ ┌─────▼──────┐
     │  PARADIGM 1:  │ │PARADIGM 2:│ │PARADIGM 3: │
     │  Direct       │ │ReAct Loop │ │MCTS Agent  │
     │  (no tools)   │ │(linear)   │ │(tree search│
     │               │ │           │ │  + backtrack│
     │  Used when:   │ │Used when: │ │Used when:  │
     │  conf > 0.8   │ │0.3<conf<0.8│ │conf < 0.3  │
     └───────────────┘ └───────────┘ └────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  FOUR-LAYER     │
                    │  MEMORY SYSTEM  │
                    │  ┌────────────┐ │
                    │  │ Working    │ │ ← KV Cache + SSM State
                    │  │ Semantic   │ │ ← Engram Memory
                    │  │ Episodic   │ │ ← EpisodicMemory (with decay)
                    │  │ Procedural │ │ ← SkillStore (executable)
                    │  └────────────┘ │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  SAFETY LAYER   │
                    │  ┌────────────┐ │
                    │  │ ToolEmu    │ │ ← Pre-execution simulation
                    │  │ Sandbox    │ │ ← Container isolation
                    │  │ Audit Log  │ │ ← Full action trace
                    │  └────────────┘ │
                    └─────────────────┘
```

### 11.2 Priority Implementation Roadmap

Based on impact/effort analysis:

| Priority | Enhancement | Source Paper | Impact | Effort |
|----------|-------------|-------------|--------|--------|
| **P0** | SMART-style knowledge sufficiency check in SignalExtractor | SMART (2025) | High | Low |
| **P0** | Multi-round retrieval with query refinement | Agentic RAG | High | Medium |
| **P1** | DFSDT-style backtracking in agent loop | ToolLLM | High | Medium |
| **P1** | Voyager-style executable skills with retry | Voyager | High | Medium |
| **P1** | Ebbinghaus forgetting in EpisodicMemory | MemoryBank | Medium | Low |
| **P1** | ToolEmu pre-execution risk assessment | ToolEmu | High | Medium |
| **P2** | LATS-style MCTS agent loop | LATS | Very High | High |
| **P2** | CREATOR two-phase tool creation | CREATOR | Medium | Medium |
| **P2** | Paradigm routing (Direct/CoT/ReAct/RAG/MCTS) | Paradigm Routing | High | High |
| **P2** | AgentTuning-style trajectory fine-tuning | AgentTuning | High | High |
| **P3** | SymbolicTranslator for formal verification | LLM→Symbolic | Medium | High |
| **P3** | DEPS failure recovery planner | DEPS | Medium | Medium |
| **P3** | Container-based terminal isolation | Best practices | Medium | Medium |

### 11.3 Key Design Principles (Synthesized from All Papers)

1. **The agent NEVER modifies the model** — it only uses model signals and feeds results back as context (already in Losion's design)

2. **Route before you act** — use Losion's Tri-Jalur signals to decide the paradigm BEFORE entering the agent loop. Don't default to ReAct for everything.

3. **Trees, not lines** — DFSDT and LATS show that tree-structured exploration (with backtracking) consistently outperforms linear loops. Use MCTS at the agent level.

4. **Abstract before concrete** — CREATOR shows that separating "what the tool should do" from "how to implement it" improves both tool quality and verification.

5. **Forget as well as remember** — MemoryBank's Ebbinghaus-inspired decay prevents memory bloat. Not all experiences are equally valuable.

6. **Emulate before execute** — ToolEmu shows that simulating tool outcomes before actual execution catches risks that static rule-checking misses.

7. **Self-improve from success** — FireAct and AgentTuning show that fine-tuning on successful agent trajectories creates better agents. Losion's EpisodicMemory is already collecting this data.

8. **Verify symbolically, reason neurally** — The neuro-symbolic pattern (Losion's NeuroSymbolicVerifier + LLM→Symbolic translation) provides formal guarantees that pure neural reasoning cannot.

---

## Appendix: Paper Reference Index

| Paper | Year | Citations | Key Concept |
|-------|------|-----------|-------------|
| Toolformer | 2023 | — | Self-supervised tool use |
| Gorilla | 2023 | — | API-augmented LLM |
| ToolLLM | 2023 | — | DFSDT for tool use |
| HuggingGPT | 2023 | — | LLM as task orchestrator |
| ReAct | 2023 | — | Interleaved reasoning + acting |
| Reflexion | 2023 | — | Self-reflection from failures |
| Voyager | 2023 | 2,174 | Skill library + lifelong learning |
| DEPS | 2023 | 447 | Describe-Explain-Plan-Select |
| GITM | 2023 | — | Sub-goal tree decomposition |
| CREATOR | 2023 | 127 | Disentangle abstract/concrete tool creation |
| LATM | 2023 | 293 | LLMs as tool makers |
| Generative Agents | 2023 | — | Memory stream + reflection |
| MemoryBank | 2023 | 790 | Ebbinghaus forgetting for agents |
| AgentVerse | 2023 | — | Multi-agent dynamic recruitment |
| FireAct | 2023 | 218 | Agent trajectory fine-tuning |
| AgentTuning | 2023 | — | Mixed agent + general instruction tuning |
| LATS | 2024 | — | MCTS for language agents (ICML 2024) |
| ToolEmu | 2024 | 326 | LM-emulated tool safety testing |
| ToolMaker | 2025 | 54 | Auto tool from code repos |
| SMART | 2025 | 37 | Tool overuse mitigation |
| A-MEM | 2025 | — | Agentic memory |
| Paradigm Routing | 2024 | — | Per-task paradigm selection |
| CRP-RAG | 2024 | 36 | Reasoning graph RAG |
| OPEN-RAG | 2024 | 79 | Self-reflective RAG |
| ProofNet++ | 2025 | — | Neuro-symbolic theorem proving |
| FVEL | 2024 | — | Formal verification with LLMs |

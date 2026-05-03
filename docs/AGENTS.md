# Losion Agent Layer — Detailed Documentation

> The Agent Layer sits **above** the Losion Tri-Jalur model, providing autonomous
> agent capabilities: skill management, tool execution, web search, reflection,
> and self-improvement. The model remains a clean neural architecture; this layer
> translates model signals into agent actions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [Module Reference](#module-reference)
   - [Signal Extraction](#signal-extraction)
   - [Orchestrator](#orchestrator)
   - [Paradigm Router](#paradigm-router)
   - [MCTS Agent Loop](#mcts-agent-loop)
   - [DEPS Planner](#deps-planner)
   - [Skills System](#skills-system)
   - [Tools System](#tools-system)
   - [Agentic Retriever](#agentic-retriever)
   - [Risk Simulator](#risk-simulator)
   - [Self-Reflection](#self-reflection)
   - [Calibration Engine](#calibration-engine)
   - [Episodic Memory](#episodic-memory)
   - [Meta-Skill System](#meta-skill-system)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Research References](#research-references)
8. [Version History](#version-history)

---

## Architecture Overview

```
                         USER QUERY
                             │
                             ▼
                    ┌─────────────────┐
                    │  LOSION MODEL   │
                    │  (Tri-Jalur)    │  ← SSM + Attention + Retrieval
                    │  + ThinkingToggle│  + MCTS + NeuroSym Verifier
                    └────────┬────────┘
                             │ model signals (confidence, routing weights, thinking mode)
                             ▼
                    ┌─────────────────┐
                    │ SIGNAL EXTRACTOR│  ← Multi-signal fusion
                    │ (SMART + Toolf.) │  + knowledge sufficiency check
                    └────────┬────────┘
                             │ AgentSignal
                             ▼
                    ┌─────────────────┐
                    │ PARADIGM ROUTER │  ← Selects: Direct | CoT | ReAct | RAG | MCTS
                    │ (5 paradigms)   │  Based on confidence + routing weights + domain
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼──────┐ ┌─────▼──────┐
     │  Direct/CoT   │ │ ReAct Loop│ │MCTS Agent  │
     │  (no tools)   │ │ (linear)  │ │(tree search│
     │               │ │           │ │ + backtrack│
     │  conf > 0.5   │ │0.3<conf   │ │ conf < 0.3 │
     └───────────────┘ │<0.5       │ └────────────┘
                       └───────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  FOUR-LAYER     │
                    │  MEMORY SYSTEM  │
                    │  ┌────────────┐ │
                    │  │ Working    │ │ ← KV Cache + SSM State (in-context, ephemeral)
                    │  │ Semantic   │ │ ← Engram Memory (model-level, persistent facts)
                    │  │ Episodic   │ │ ← EpisodicMemory (experiences + Ebbinghaus decay)
                    │  │ Procedural │ │ ← SkillStore (executable skills with retry logic)
                    │  └────────────┘ │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  SAFETY LAYER   │
                    │  ┌────────────┐ │
                    │  │ ToolEmu    │ │ ← Pre-execution risk simulation
                    │  │ Sandbox    │ │ ← Container isolation, command validation
                    │  │ Audit Log  │ │ ← Full action trace
                    │  └────────────┘ │
                    └─────────────────┘
```

### Directory Structure

```
losion/agent/
├── __init__.py              # Public API exports (all agent components)
├── signals.py               # Bridge: model output → agent decisions
├── orchestrator.py          # Central coordinator: agent loop + paradigm routing
├── reflection.py            # Self-reflection engine (Reflexion + Self-Refine)
├── calibration.py           # Adaptive confidence calibration (ATTC)
├── memory.py                # Episodic memory with Ebbinghaus forgetting
├── meta_skills.py           # Meta-skill system: synthesis, verification, composition
├── planning/
│   ├── __init__.py
│   ├── paradigm_router.py   # Routes queries to reasoning paradigms
│   ├── mcts_agent.py        # LATS-style MCTS agent loop
│   └── deps_planner.py      # DEPS failure recovery planner
├── retrieval/
│   ├── __init__.py
│   └── agentic_retriever.py # Multi-round retrieval with query refinement
├── safety/
│   ├── __init__.py
│   └── risk_simulator.py    # ToolEmu-style pre-execution risk assessment
├── skills/
│   ├── __init__.py
│   ├── store.py             # Persistent skill storage (Engram-like)
│   ├── manager.py           # Skill registry + lookup + auto-create
│   └── creator.py           # Auto-generate skills from web search
└── tools/
    ├── __init__.py
    ├── registry.py           # Tool registry + discovery
    ├── terminal.py           # Terminal execution (sandboxed)
    ├── web_search.py         # Web search interface
    └── creator.py            # Auto-generate tools when missing
```

---

## Design Principles

1. **Agent Layer is separate from neural architecture** — The model provides signals; the agent responds. No agent code modifies model weights or architecture.

2. **Model signals → Agent actions** — ThinkingToggle confidence, Tri-Jalur routing weights, and thinking mode are translated into agent signals via `SignalExtractor`.

3. **Skills & Tools stored externally** — Not in model weights. SkillStore uses Engram-like hash-based storage. Tools are registered in ToolRegistry with safety classifications.

4. **Terminal execution in isolated sandbox** — SandboxedTerminal validates commands with blocked patterns, resource limits, and timeouts.

5. **Web search only on low confidence** — SMART-style knowledge sufficiency check prevents tool overuse when parametric knowledge is sufficient.

6. **Everything is optional and configurable** — Every agent feature can be disabled via `AgentConfig`. The model works perfectly without the agent layer.

7. **Learn from experience** — ReflectionEngine generates verbal feedback. CalibrationEngine adapts thresholds. EpisodicMemory stores and retrieves past experiences.

8. **Tree-structured action exploration** — MCTS Agent Loop replaces linear iteration with tree search, backtracking, and simulation (LATS + DFSDT).

9. **Pre-execution safety** — RiskSimulator assesses risk before executing dangerous commands (ToolEmu-style).

---

## Module Reference

### Signal Extraction

**File:** `losion/agent/signals.py`

The bridge between Losion's model output and the agent layer. It translates model-internal signals into actionable agent signals.

**Key classes:**
- `AgentAction` — Enum of 9 possible actions: `MODEL_ONLY`, `WEB_SEARCH`, `SKILL_LOOKUP`, `SKILL_CREATE`, `TOOL_SEARCH`, `TOOL_CREATE`, `TERMINAL_EXECUTE`, `VERIFY_OUTPUT`, `REFLECT`
- `AgentSignal` — A signal with action, confidence, reasoning, query, domain, priority, tool_trust, knowledge_sufficient flag, and perplexity signal
- `ConfidenceThreshold` — Configurable thresholds for each action type
- `SignalExtractor` — The main extractor that fuses multiple signals

**Signal fusion process:**
1. **Confidence signal** — Low confidence triggers external action
2. **Routing signal** — High retrieval weight (Jalur 3) triggers search/lookup
3. **Thinking signal** — Thinking mode + low confidence triggers deep intervention
4. **Task type signal** — Factual task + low confidence triggers web search
5. **Experience signal** (v2) — Past episodes recommend specific actions
6. **Trust signal** (v2) — Tool trust influences action priority
7. **Knowledge sufficiency** (v3) — SMART check prevents tool overuse
8. **Perplexity signal** (v3) — Toolformer-inspired token-level confidence

**SMART Knowledge Sufficiency Check:**
```python
# If retrieval weight is LOW and attention weight is HIGH and confidence is decent
# then parametric knowledge is sufficient — no tool use needed
knowledge_sufficient = (
    routing_weights[2] < 0.2    # Model doesn't need retrieval
    and routing_weights[1] > 0.4  # Reasoning pathway is active
    and confidence > 0.5           # Model is somewhat confident
)
```

---

### Orchestrator

**File:** `losion/agent/orchestrator.py`

The central coordinator that manages the agent loop. It receives model output, extracts signals, routes to the appropriate paradigm, executes actions, reflects on outcomes, and calibrates thresholds.

**Key classes:**
- `AgentConfig` — Full configuration with v1/v2/v3 options
- `AgentResult` — Complete result with output, actions, reflections, episode, metadata
- `AgentOrchestrator` — The main orchestrator class

**Agent loop (v2/v3):**
1. Extract signal from model output (with adaptive thresholds)
2. If `MODEL_ONLY` → return current output
3. If `REFLECT` → load reflections from episodic memory
4. Route to appropriate paradigm via ParadigmRouter
5. Execute action (with risk assessment for terminal commands)
6. Reflect on outcome (Reflexion + Self-Refine)
7. Calibrate thresholds based on outcome (ATTC)
8. Inject context from action result
9. Re-infer with new context (if model_inference_fn provided)
10. Store episode in episodic memory
11. Repeat until confidence sufficient or max iterations

**v3 components initialized by the orchestrator:**
- `ParadigmRouter` — Routes to Direct/CoT/ReAct/RAG/MCTS
- `MCTSAgentLoop` — Tree-structured action exploration
- `DEPSPlanner` — Structured failure recovery
- `AgenticRetriever` — Multi-round web search
- `RiskSimulator` — Pre-execution risk assessment

---

### Paradigm Router

**File:** `losion/agent/planning/paradigm_router.py`

The agent-level equivalent of Losion's Tri-Jalur Router. While the Tri-Jalur Router selects between SSM, Attention, and Retrieval at the model level, the Paradigm Router selects between five reasoning paradigms at the agent level.

**Five paradigms (lightest to heaviest):**

| Paradigm | Confidence Range | Uses Tools? | Cost | When to Use |
|----------|-----------------|-------------|------|-------------|
| DIRECT | > 0.8 | No | 1 | Model is confident, parametric knowledge sufficient |
| COT | 0.5 – 0.8 | No | 2 | Needs reasoning but no external information |
| REACT | 0.3 – 0.5 | Yes | 3 | Interleaved thought-action-observation needed |
| RAG | 0.15 – 0.3 | Yes | 3 | Single retrieval + generation sufficient |
| MCTS | < 0.15 | Yes | 5 | Complex, needs tree search with backtracking |

**Routing adjustments:**
- SMART knowledge sufficiency → downgrades tool-using paradigms when parametric knowledge is sufficient
- Domain adjustments → Math prefers CoT, Code prefers REACT, History prefers RAG
- Task type adjustments → Factual prefers RAG/Direct, Reasoning prefers REACT
- Thinking mode override → If model is already thinking, upgrade to at least CoT
- Calibration adjustments → If tools are unreliable for a domain, prefer lighter paradigms

---

### MCTS Agent Loop

**File:** `losion/agent/planning/mcts_agent.py`

Replaces the linear agent loop with LATS-style tree-structured action exploration. This enables the agent to explore multiple action sequences, backtrack when actions fail, and select the best path through the decision tree.

**MCTS Cycle:**
1. **SELECT** — UCB1 traversal to find the most promising leaf node
2. **EXPAND** — Generate possible actions using SignalExtractor
3. **SIMULATE** — Execute the action and observe the result
4. **BACKTRACK** (DFSDT-style) — If confidence drops significantly, mark node as failed and try alternatives
5. **BACKPROPAGATE** — Update values up the tree with discount factor

**Key data structures:**
- `AgentState` — Query, context, confidence, actions_taken, domain, routing_weights
- `ActionNode` — Tree node with UCB1 scoring, visits, value, status, parent/children links
- `ActionEdge` — Records action taken, result, confidence before/after, tool trust, reward
- `MCTSResult` — Best path, final state, simulation count, backtracks, tree size

**Reward computation:**
```python
reward = confidence_delta * tool_trust_modifier * novelty_bonus
```

**UCB1 formula:**
```python
UCB1 = Q(s,a) + C * sqrt(ln(N(parent)) / N(s,a))
# C = sqrt(2), standard exploration constant
```

---

### DEPS Planner

**File:** `losion/agent/planning/deps_planner.py`

Structured failure recovery via Describe-Explain-Plan-Select. When an agent action fails, the DEPS Planner provides a principled recovery strategy instead of simple retry.

**Four phases:**
1. **DESCRIBE** — Classify the failure type and capture what happened
2. **EXPLAIN** — Generate an explanation of why the failure occurred
3. **PLAN** — Generate multiple alternative approaches with estimated success rates
4. **SELECT** — Choose the best plan based on success rate, simplicity, and experience

**Seven failure types:**
- `TOOL_FAILURE` — Tool execution failed
- `SEARCH_FAILURE` — Web search returned no results
- `SKILL_NOT_FOUND` — No suitable skill exists
- `CONFIDENCE_DROP` — Action reduced confidence
- `TIMEOUT` — Action timed out
- `VALIDATION_FAILURE` — Output verification failed
- `CONTEXT_INSUFFICIENT` — Retrieved context not enough

**Recovery strategies** are pre-defined for each failure type, with:
- Multiple alternative plans ranked by estimated success rate
- Fallback chains (plan A → plan B → plan C)
- Experience-adjusted success rates from episodic memory

---

### Skills System

**Files:** `losion/agent/skills/`

A three-component skill management system:

**SkillStore** (`store.py`) — Persistent storage for skill definitions.
- Hash-based O(1) lookup by name
- Tag-based search for skill discovery
- Domain-based filtering
- Usage tracking (times_used, last_used, success_rate)
- Disk persistence at `~/.losion/skills/`

**SkillManager** (`manager.py`) — Registry + lookup + auto-create.
- `lookup(query, domain, tags)` → Searches store by name, tags, and domain
- Auto-creates skills when lookup fails (if enabled)
- Records usage statistics for calibration
- Composes multiple skills when no single skill matches

**SkillCreator** (`creator.py`) — Auto-generates skills from web search.
- Searches the web for relevant information
- Generates skill definition from search context
- Tags the skill with domain and keywords
- Stores in SkillStore for future use

**SkillEntry structure:**
```python
@dataclass
class SkillEntry:
    name: str                    # Unique skill name
    definition: str              # Skill description and instructions
    metadata: SkillMetadata      # Confidence, domain, tags, source, usage stats
```

---

### Tools System

**Files:** `losion/agent/tools/`

**ToolRegistry** (`registry.py`) — Tool registry with safety classifications.
- Register tools with name, description, handler, safety level, domain, tags
- Search tools by query, domain, or tags
- Three safety levels: `SAFE`, `REQUIRES_APPROVAL`, `DANGEROUS`
- Builtin tools: `web-search`, `terminal`

**SandboxedTerminal** (`terminal.py`) — Terminal execution in an isolated sandbox.
- Blocked commands list (rm -rf /, mkfs, format, etc.)
- Blocked patterns (fork bombs, pipe to shell, etc.)
- Execution timeout (default 30s)
- Working directory restriction
- Output size limit
- Full audit logging

**WebSearchInterface** (`web_search.py`) — Web search interface.
- Configurable search backend
- Result validation and deduplication
- Relevance scoring
- Source tracking

**ToolCreator** (`creator.py`) — Auto-generates tools when missing.
- Creates tools based on query and domain
- Auto-registers in ToolRegistry
- Compatible with CREATOR's two-phase approach (abstract spec → concrete implementation)

---

### Agentic Retriever

**File:** `losion/agent/retrieval/agentic_retriever.py`

Multi-round retrieval with confidence-based query refinement. Unlike single-round search, this retriever iteratively refines the query until results are sufficient.

**Process:**
1. Initial search with user's query
2. Assess result quality (count, relevance, coverage, richness)
3. If insufficient, refine the query using one of five strategies
4. Re-search with refined query
5. Synthesize results across all rounds (deduplicate, re-rank, merge)

**Query refinement strategies:**
- `ADD_CONTEXT` — Add domain-specific terms from partial results
- `REPHRASE` — Rephrase using alternative framing ("how to" ↔ "guide for")
- `DECOMPOSE` — Split complex queries into focused sub-queries
- `NARROW` — Add specific terms to narrow broad results
- `BROADEN` — Remove specific terms to broaden narrow results

**Quality assessment composite score:**
```
quality = count_score * 0.2 + relevance * 0.3 + coverage * 0.3 + richness * 0.2
```

---

### Risk Simulator

**File:** `losion/agent/safety/risk_simulator.py`

ToolEmu-style pre-execution risk assessment. Before executing any potentially dangerous action, the simulator assesses the risk level and routes to appropriate execution path.

**Three-layer assessment:**
1. **Static analysis** — Pattern matching against known dangerous commands (CRITICAL, HIGH, MEDIUM patterns), protected file paths, network operations
2. **Dynamic simulation** — Heuristic-based outcome prediction (rm → data loss, curl → malicious code download, pip → package risks)
3. **Experience-based assessment** — Checks episodic memory for past failures with similar actions

**Five risk levels:**
| Level | Action | Approval Required |
|-------|--------|-------------------|
| SAFE | Execute immediately | No |
| LOW | Execute with logging | No |
| MEDIUM | Request approval | Yes |
| HIGH | Require approval with audit | Yes |
| CRITICAL | Block execution | Always blocked |

**Calibration integration:** If CalibrationEngine reports low tool trust, risk level is increased by one step.

---

### Self-Reflection

**File:** `losion/agent/reflection.py`

Reflexion + Self-Refine inspired self-evaluation engine. After each agent action, the engine assesses the outcome and generates structured verbal feedback.

**Six reflection types:**
- `ACTION_SUCCESS` — Action produced useful results
- `ACTION_FAILURE` — Action did not produce expected results
- `STRATEGY_CORRECTION` — Current strategy is ineffective, suggesting alternatives
- `TOOL_TRUST_UPDATE` — Tool reliability assessment based on outcome
- `SKILL_REFINEMENT` — Suggestion to improve an existing skill
- `CONFIDENCE_RECALIBRATION` — Confidence should be adjusted

**Assessment heuristics:**
1. Result presence (None = failure)
2. Confidence change (positive delta = success)
3. Content analysis (success/failure indicator keywords)
4. Structured result fields (dict with "success" key)

**Generated lessons** are stored in EpisodicMemory and retrieved by SignalExtractor for future similar queries.

---

### Calibration Engine

**File:** `losion/agent/calibration.py`

Adaptive confidence calibration that replaces static thresholds with experience-based, domain-specific, tool-trust-aware thresholds.

**Three calibration signals:**
1. **Domain Profiles** — 7 pre-defined domain profiles with different optimal thresholds
   - Math: Higher web search threshold (model is usually right)
   - Code: Lower tool search threshold (tools are very helpful)
   - History: Lower web search threshold (facts need verification)
   - Web: Lower tool search threshold (API tools critical)
   - Data: Lower terminal threshold (terminal useful for data)

2. **Tool Trust Scores** — EMA-based reliability tracking per tool per domain
   ```python
   trust = alpha * new_evidence + (1 - alpha) * old_trust
   # alpha = 0.3, moderate adaptation speed
   ```

3. **Outcome History** — Per-domain, per-action records of successes/failures
   - Successful actions that improved confidence → lower thresholds (use more eagerly)
   - Failed actions that reduced confidence → raise thresholds (use more cautiously)
   - Conservative learning rate prevents overfitting to recent outcomes

---

### Episodic Memory

**File:** `losion/agent/memory.py`

Experience-based memory with Ebbinghaus forgetting curve and multi-factor retrieval. Part of the four-layer memory architecture.

**Four memory layers:**
| Layer | Storage | Persistence | Content |
|-------|---------|-------------|---------|
| Working | KV Cache + SSM State | Ephemeral | Current context |
| Semantic | Engram Memory | Persistent | Facts, knowledge |
| Episodic | This module | Persistent | Past experiences |
| Procedural | SkillStore | Persistent | How to do things |

**Ebbinghaus forgetting curve:**
```python
decay = exp(-0.1 * age_in_days)              # Exponential forgetting
reinforcement = 1.0 + 0.2 * log(1 + access_count)  # Access strengthens memory
effective_strength = strength * decay * reinforcement
```

**Multi-factor retrieval (Generative Agents style):**
```
composite = recency * importance * relevance * effective_strength
```
- **Recency**: `exp(-0.05 * age_hours)` — half-life ~14 hours
- **Importance**: Based on confidence and success (successful episodes × 1.2, failed × 0.6)
- **Relevance**: Jaccard similarity + domain bonus/penalty
- **Effective strength**: Ebbinghaus curve with reinforcement

**Spreading Activation (Synapse):** When an episode is retrieved, related episodes in the same domain are also activated with decaying strength.

**Periodic Consolidation:**
1. Remove episodes with effective strength below threshold (default 0.05)
2. Merge very similar episodes (Jaccard > 0.8, same domain) — keep the stronger one

---

### Meta-Skill System

**File:** `losion/agent/meta_skills.py`

Three meta-skills that enable the agent to improve its own skill capabilities:

**1. SkillSynthesisMetaSkill** — How to create skills effectively.
- Generates multiple search queries for the same task (different framings)
- Aggregates and deduplicates search results
- Cross-references for consistency
- Generates test cases alongside skill definitions
- Higher initial confidence for meta-synthesized skills (multiple sources + tests)

**2. SkillVerificationMetaSkill** — How to test and validate skills.
- Extracts test cases from skill definition (## Test Cases section)
- Runs test cases against skill definition
- Bayesian confidence update: `confidence = (1 - w) * prior + w * evidence`
- Generates recommendations for improving failing skills

**3. SkillCompositionMetaSkill** — How to chain skills together.
- Decomposes complex queries into sub-tasks (conjunction-based splitting)
- Finds skills for each sub-task
- Checks compatibility between skill outputs/inputs
- Composes into ordered pipelines with metadata
- Identifies missing skills and recommends creating them

---

## Data Flow

### Complete Agent Loop Flow

```
1. User Query
   │
   ▼
2. Model Inference (Tri-Jalur)
   │ → routing_weights = [w_ssm, w_attn, w_retr]
   │ → thinking_assessment = {mode, confidence, dominant_task}
   │ → model_output
   │
   ▼
3. Signal Extraction
   │ → Extract: thinking_mode, routing_weights, confidence, task_type, domain
   │ → SMART check: is parametric knowledge sufficient?
   │ → Fuse signals: confidence + routing + thinking + task + experience + trust
   │ → Output: AgentSignal {action, confidence, priority, query, domain}
   │
   ▼
4. Paradigm Routing
   │ → confidence → baseline paradigm
   │ → SMART check → downgrade if knowledge sufficient
   │ → domain adjustments → math→CoT, code→REACT, history→RAG
   │ → task type adjustments → factual→RAG, reasoning→REACT
   │ → calibration adjustments → unreliable tools → lighter paradigms
   │ → Output: ParadigmSelection {paradigm, alternatives, reasoning}
   │
   ▼
5. Execute Paradigm
   │
   ├─ DIRECT: Return model output (no tools)
   │
   ├─ COT: Return model output with thinking context
   │
   ├─ REACT: Linear agent loop
   │   │ → Execute action (web_search/skill_lookup/tool_search/terminal/verify)
   │   │ → Reflect on outcome (ReflectionEngine)
   │   │ → Calibrate thresholds (CalibrationEngine)
   │   │ → Inject context, re-infer
   │   │ → Repeat until confidence ≥ 0.7 or max iterations
   │
   ├─ RAG: Single retrieval + generation
   │   │ → Execute one web search (AgenticRetriever if available)
   │   │ → Inject context, re-infer
   │
   └─ MCTS: Tree-structured exploration
       │ → SELECT (UCB1 traversal)
       │ → EXPAND (generate candidate actions)
       │ → SIMULATE (execute action, compute reward)
       │ → BACKTRACK (if confidence drops > threshold)
       │ → BACKPROPAGATE (update values up tree)
       │ → Repeat for max_simulations
       │ → Return best path
   │
   ▼
6. On Failure: DEPS Recovery
   │ → DESCRIBE: What happened? (FailureDescription)
   │ → EXPLAIN: Why did it fail? (root cause analysis)
   │ → PLAN: Generate alternatives (RecoveryPlan with success rates)
   │ → SELECT: Choose best plan (success rate + simplicity + experience)
   │
   ▼
7. Post-Action
   │ → Risk assessment for dangerous actions (RiskSimulator)
   │ → Reflection on outcome (ReflectionEngine)
   │ → Calibration update (CalibrationEngine)
   │ → Episode storage (EpisodicMemory with Ebbinghaus decay)
   │
   ▼
8. Return AgentResult
     {output, state, iterations, actions_taken, signals, context_used,
      skills_used, tools_used, reflections, episode_id, model_confidence}
```

---

## Configuration

### AgentConfig Reference

```python
@dataclass
class AgentConfig:
    # === v1: Core Settings ===
    max_iterations: int = 5                    # Max agent loop iterations
    enable_web_search: bool = True             # Web search available
    enable_terminal: bool = True               # Terminal execution available
    enable_skill_creation: bool = True         # Auto-create skills
    enable_tool_creation: bool = True          # Auto-create tools
    confidence_thresholds: ConfidenceThreshold  # Per-action thresholds
    auto_inject_context: bool = True           # Auto-inject search results
    sandbox_config: SandboxConfig              # Terminal sandbox settings
    skill_store_dir: str = "~/.losion/skills"  # Skill storage directory
    verbose: bool = False                      # Detailed logging

    # === v2: Self-Improvement ===
    episodic_store_dir: str = "~/.losion/episodic"  # Episode storage
    enable_reflection: bool = True             # Self-reflection engine
    enable_calibration: bool = True            # Adaptive thresholds
    enable_meta_skills: bool = True            # Meta-skill system
    reflection_on_failure: bool = True         # Reflect on action failure
    calibration_learning_rate: float = 0.1     # How fast thresholds adapt

    # === v3: Advanced Features ===
    enable_paradigm_routing: bool = True       # Paradigm router
    enable_mcts_agent: bool = True             # MCTS agent loop
    enable_deps_recovery: bool = True          # DEPS failure recovery
    enable_risk_simulation: bool = True        # Risk simulator
    enable_agentic_retrieval: bool = True      # Multi-round retrieval
    mcts_max_simulations: int = 8             # MCTS simulation budget
    mcts_max_depth: int = 5                    # Max MCTS tree depth
    risk_threshold: str = "medium"             # Risk tolerance level
```

### ConfidenceThreshold Reference

```python
@dataclass
class ConfidenceThreshold:
    web_search: float = 0.3      # Trigger web search below this
    skill_lookup: float = 0.4    # Trigger skill lookup below this
    tool_search: float = 0.35    # Trigger tool search below this
    verify: float = 0.5          # Trigger verification below this
    terminal: float = 0.25       # Trigger terminal below this
```

---

## Usage Examples

### Basic Usage

```python
from losion.agent import AgentOrchestrator, AgentConfig

# Create orchestrator with default config
orchestrator = AgentOrchestrator()

# Run with model output
result = orchestrator.run(
    model_output=routing_output,  # From AdaptiveRouter.forward()
    query="What is the capital of France?",
    confidence=0.2,                # Low confidence → agent intervention
    model_inference_fn=my_model_fn,  # For re-inference with context
)

print(result.output)
print(f"Actions taken: {result.actions_taken}")
print(f"Final confidence: {result.model_confidence:.2f}")
print(f"Episodes stored: {result.episode_id}")
```

### With Custom Configuration

```python
config = AgentConfig(
    max_iterations=10,
    enable_web_search=True,
    enable_terminal=True,
    enable_skill_creation=True,
    enable_reflection=True,
    enable_calibration=True,
    enable_paradigm_routing=True,
    enable_mcts_agent=True,
    enable_risk_simulation=True,
    mcts_max_simulations=16,
    risk_threshold="high",  # Conservative risk tolerance
    verbose=True,
)

orchestrator = AgentOrchestrator(config)
result = orchestrator.run(
    query="Solve this differential equation: dy/dx = x^2 + y",
    model_output=routing_output,
    confidence=0.15,  # Very low → MCTS paradigm
)
```

### Using Individual Components

```python
from losion.agent import SignalExtractor, ConfidenceThreshold
from losion.agent.planning import ParadigmRouter, ReasoningParadigm

# Use signal extractor independently
extractor = SignalExtractor(thresholds=ConfidenceThreshold(web_search=0.4))
signal = extractor.extract(
    model_output=routing_output,
    confidence=0.3,
    query_text="How does quantum entanglement work?",
)

# Use paradigm router independently
router = ParadigmRouter()
selection = router.route(
    confidence=0.3,
    routing_weights=[0.3, 0.5, 0.2],
    query="How does quantum entanglement work?",
    domain="science",
)
print(f"Selected paradigm: {selection.paradigm.value}")
```

### Episodic Memory Management

```python
from losion.agent import EpisodicMemory, Episode

# Create memory store
memory = EpisodicMemory(store_dir="~/.losion/episodic")

# Store an episode
episode = Episode(
    query="Calculate the integral of x^2",
    domain="math",
    actions=["skill_lookup", "terminal_execute"],
    reflections=[{"lesson": "Math queries benefit from terminal computation"}],
    final_confidence=0.85,
    success=True,
)
memory.store_episode(episode)

# Retrieve lessons for a new query
lessons = memory.get_lessons_for_query(
    query="Integrate sin(x) from 0 to pi",
    domain="math",
)
# → ["Math queries benefit from terminal computation"]

# Periodic consolidation
stats = memory.consolidate(strength_threshold=0.05, similarity_threshold=0.8)
# → {"removed_weak": 3, "merged": 2, "remaining": 45}
```

### Risk Assessment

```python
from losion.agent.safety import RiskSimulator, RiskLevel

simulator = RiskSimulator(risk_threshold=RiskLevel.MEDIUM)

# Assess a terminal command
assessment = simulator.assess(
    action="terminal_execute",
    command="rm -rf /tmp/old_logs",
    domain="system",
)

if assessment.should_execute:
    if assessment.needs_approval:
        if get_user_approval(assessment.reasoning):
            terminal.execute("rm -rf /tmp/old_logs")
    else:
        terminal.execute("rm -rf /tmp/old_logs")
else:
    print(f"BLOCKED: {assessment.reasoning}")
```

---

## Research References

The Losion Agent Layer is built on insights from 40+ research papers. Full citations are in [CREDITS.md](../CREDITS.md). Key references by category:

### Tool Use & API Calling
- **Toolformer** (Schick et al., 2023) — Self-supervised tool use via perplexity filtering
- **Gorilla** (Patil et al., 2023) — Fine-tuning for API calls with AST evaluation
- **ToolLLM** (Qin et al., 2023) — DFSDT for tree-structured tool exploration
- **HuggingGPT** (Shen et al., 2023) — LLM as orchestrator pipeline

### Agentic Frameworks
- **ReAct** (Yao et al., 2023) — Interleaved reasoning and acting
- **Reflexion** (Shinn et al., 2023) — Learning from verbal feedback
- **Self-Refine** (Madaan et al., 2023) — Iterative self-improvement
- **AutoGPT/BabyAGI** (2023) — Autonomous task decomposition

### Skill & Tool Creation
- **Voyager** (Wang et al., 2023, cited 2,174×) — Executable skill library
- **CREATOR** (Qian et al., 2023, cited 127×) — Two-phase tool creation
- **LATM** (Cai et al., 2023, cited 293×) — LLMs as tool makers
- **CASCADE** (2025) — Cumulative agentic skill creation

### Planning & Search
- **LATS** (Zhou et al., ICML 2024) — Language Agent Tree Search
- **DEPS** (Wang et al., 2023, cited 447×) — Describe-Explain-Plan-Select
- **GITM** (Zhu et al., 2023) — Sub-goal tree decomposition

### Memory Systems
- **Generative Agents** (Park et al., 2023) — Multi-factor retrieval
- **MemoryBank** (Zhong et al., 2023, cited 790×) — Ebbinghaus forgetting curve
- **Synapse** (2024) — Spreading activation memory

### Safety & Calibration
- **ToolEmu** (Ruan et al., ICLR 2024 Spotlight, cited 326×) — Risk emulation
- **SMART** (2025) — Tool overuse mitigation
- **ATTC** (2026) — Adaptive tool trust calibration

### Self-Improvement
- **FireAct** (Chen et al., 2023, cited 218×) — Agent trajectory fine-tuning
- **AgentTuning** (Zeng et al., 2023) — Mixed agent/general training

---

## Version History

### v3 — Research-Informed Improvements (Current)
- LATS-style MCTS Agent Loop with tree-structured action exploration
- SMART Knowledge Sufficiency Check preventing tool overuse
- Paradigm Router with 5 reasoning paradigms
- DEPS Failure Recovery with structured Describe→Explain→Plan→Select
- Agentic Multi-round Retrieval with query refinement
- ToolEmu Risk Simulator for pre-execution assessment
- Ebbinghaus Memory Decay with access reinforcement
- Multi-factor Retrieval (recency × importance × relevance × strength)
- Perplexity-based confidence estimation (Toolformer-inspired)

### v2 — Self-Improvement
- Reflexion-inspired Self-Reflection Engine
- ATTC Adaptive Confidence Calibration
- Episodic Memory for experience-based decisions
- Meta-Skill System (synthesis, verification, composition)
- Domain-specific calibration profiles
- Tool trust scores with EMA updates

### v1 — Foundation
- Signal Extraction bridge (model → agent)
- Agent Orchestrator with linear agent loop
- SkillStore, SkillManager, SkillCreator
- ToolRegistry, ToolCreator
- SandboxedTerminal with command validation
- WebSearchInterface
- Model-only inference when confidence is high

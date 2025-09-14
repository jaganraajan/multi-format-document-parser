# ADR 0001: Hybrid, Configurable Parsing Pipeline vs Monolithic LLM

Status: Accepted  
Date: 2025-09-14  
Owner: multi-format-document-parser maintainers  
Related Epic / Story: Multi‑Format Document Parser (cost-efficient normalization at scale)

## 1. Context
We ingest heterogeneous documents (clean PDFs, noisy scans, forwarded emails, exported HTML, text fragments). Downstream consumers require a single, stable normalized JSON schema. A naïve architecture would simply call a large language model (LLM) for every document and let it infer structure. While fast to prototype, that approach becomes:
- Economically volatile (cost scales linearly with volume; no amortization)
- Operationally opaque (hard to debug extraction drift or regression)
- Difficult to *stabilize* (model upgrades or prompt tweaks shift semantics globally)
- Hard to offer *predictable SLAs* (LLM latency variance, rate limits, vendor incidents)
- Brittle for compliance / audits (no transparent intermediate reasoning trace)

Instead, this project adopts a **hybrid, configurable pipeline** that composes:
1. Deterministic content + layout extraction (format‑specific extractors)
2. Layout signature learning & reuse (per-sender / global structural fingerprints)
3. Rule-based extraction (regex / lightweight patterns; optional overrides per signature)
4. Selective, gated AI assistance (LLM + Azure Document Intelligence) used sparingly
5. Versioned decision logic (pipeline version + signature version for reproducibility)
6. Confidence & coverage metrics enabling cost-aware routing

## 2. Decision
Adopt and evolve a **hybrid pipeline** where *rules + signatures* are first-class citizens, and *AI is invoked only when necessary* (bootstrap missing fields, handle outliers, or improve recall in low-confidence zones). The pipeline exposes user-toggles (LLM, DI) and internal gating logic:
- If rule + signature extraction yields high confidence & required coverage → skip AI
- If critical fields missing or confidence below threshold → call LLM (primary)
- If LLM yields zero fields for eligible PDFs and DI is enabled → fallback to Document Intelligence
- Optional sampling of borderline cases for continuous improvement without full-volume AI cost

## 3. Why This Is Superior to a Monolithic LLM Approach
| Dimension | Hybrid Pipeline | Monolithic LLM |
|-----------|-----------------|----------------|
| Cost Predictability | AI usage is *conditional*; unit cost amortizes downward as signatures/rules accumulate | Cost scales linearly with every document; little amortization |
| Interpretability | Transparent: signature match scores, rules applied, per-field confidence, logs | Opaque: one big response blob; hard to attribute errors |
| Drift Resistance | Versioned rules & signatures; incremental evolution | Model/prompt changes cause global semantic shifts |
| Determinism | Rule hits stable; only unknown deltas go to AI | Stochastic variation unless temperature forced low |
| Scaling Behavior | Early-stage higher AI %, declines over time | Flat high AI % forever |
| Governance / Audit | Traceable: which layer produced each field | Hard to reconstruct reasoning |
| Latency | Fast-path local extraction for common layouts | Always LLM round trip |
| Failure Isolation | Rule/sig layers continue if AI vendor down | Hard outage if LLM unavailable |
| Vendor Flexibility | AI layer pluggable and minimized | Deep coupling to one LLM provider |

## 4. Forces / Requirements Driving the Decision
- Growing volume of semi-repetitive business documents where structural patterns *stabilize* quickly.
- Need for **stable JSON schema** unaffected by underlying model version changes.
- Need to **control marginal cost** as volume scales (board / finance sensitivity).
- Requirement for **explanation artifacts** (processing log, signature reuse stats, extraction method provenance).
- Desire to **gradually learn** new senders' layouts without expensive re-processing of historical corpus.

## 5. Architecture Overview (Layers)
1. Extraction Layer: Format-specific (PDF/Text/Email) content + coarse layout elements.
2. Signature Layer: Quantized layout tokens → Jaccard similarity → reuse or create signature; track version + sample filenames.
3. Rule Layer: Global + signature-specific rule sets; provides deterministic fields with base confidences.
4. Gating & Confidence Layer: Aggregates per-field confidence (rule confidences, future learned reliability scores) → document-level confidence score; evaluates required coverage.
5. AI Layer (Conditional): LLM extraction (key fields JSON) + optional DI fallback for PDFs.
6. Repository & Analytics: Persist normalized document + meta (coverage, match score, model calls, cost placeholder) for monitoring.

## 6. Alternatives Considered
### A. Always-LLM (Monolithic)
Pros: Minimal upfront engineering; fast initial demo.  
Cons: High recurring cost, opaque errors, uncontrollable drift, no amortization, unpredictable latency.

### B. Rules-Only
Pros: Deterministic, cheap.  
Cons: Poor recall on noisy/unseen layouts; brittle to novel formatting; expensive to author/maintain manually.

### C. Embedding Similarity + LLM Only
Pros: Some layout clustering; potential reduction in prompt size.  
Cons: Still sends most docs through AI; limited structural interpretability; drift issues persist.

Hybrid selected as it combines determinism (rules/signatures) with flexible generalization (LLM/DI) and explicit gating to *shrink* AI surface area over time.

## 7. Trade-offs
- Added implementation complexity (signature store, rule engine, gating logic) vs. simpler prototype.
- Need ongoing curation of confidence thresholds and rule evolution.
- Slight initial latency overhead for signature matching (negligible vs. LLM round trip).

## 8. Consequences (Positive)
- Decreasing marginal cost curve as corpus matures.
- Easier RCA (root cause analysis) via layered logs.
- Enables AB testing of new AI models confined to low-confidence cases.
- Facilitates compliance (clear attribution: rule vs model).

## 9. Consequences (Negative / Risks)
- Over-aggressive gating could under-call AI and miss recoverable fields → mitigated by sampling borderline documents.
- Signature explosion if similarity threshold poorly tuned → mitigated via periodic consolidation.
- Stale rules may persist; need versioning criteria and retirement policy.

## 10. Cost & Confidence Model (Planned Enhancements)
Planned evolution (partially scaffolded, future PRs):
- Per-field dynamic confidence recalibrated via historical precision/recall metrics per extraction method.
- Document confidence = f(required field coverage, mean field confidence, signature match score).
- Adaptive budget: Set max model calls per N documents; prioritize lowest-confidence subset.

## 11. Versioning & Stability Strategy
- `processing_meta.pipeline_version` stored per document.
- Signatures carry their own `version`; changes (e.g., tokenization strategy) create new version while retaining old for reproducibility.
- Rule sets versioned via file-level semantic version + changelog (future enhancement) enabling audit diffs.

## 12. Observability Hooks
- Model calls counted (`model_calls_made`).
- Coverage stats: rule coverage vs model coverage for cost trend monitoring.
- Signature reuse metrics: avg documents per signature → indicator of amortization efficiency.

## 13. Future Evolution
| Area | Roadmap Idea |
|------|--------------|
| Confidence Engine | Calibrate scores using holdout validation set |
| Active Learning | Auto-queue low-confidence fields for human validation |
| Rule Synthesis | Generate candidate regex from consistent LLM outputs to lock in stability |
| Cost Attribution | Add token usage + pricing to `total_cost_usd` |
| Layout Vectorization | Optional embedding-based pre-clustering before signature match |
| Drift Alerts | Detect sudden drop in match score distribution for a high-volume signature |

## 14. Decision Summary
A hybrid, gated architecture is adopted to achieve: predictable cost, high interpretability, graceful scaling, controlled drift, and incremental learning—benefits unattainable in a simple monolithic LLM pipeline. AI becomes a *targeted accelerator* rather than a hard dependency.

---

## 15. References
- Internal module docs: `src/normalization/pipeline.py`, `signatures/`, `rules/`
- Prior art: RAG hybrid extraction patterns; cost-aware inference gating patterns in production ML systems.
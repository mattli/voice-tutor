---
created_at: 2026-04-09
last_updated: 2026-04-20
---

# Vertical AI

> TLDR: Domain-specific AI applications are finding product-market fit faster than horizontal AI, especially in legal, healthcare, and coding. Harvey AI's $0→$200M ARR in 36 months is the benchmark case study for how vertical AI companies win.

## Enterprise AI Adoption (2026 Data)

Per a16z analysis (Apr 2026), based on internal data and executive conversations:

- **29% of Fortune 500** and **~19% of Global 2000** are live, paying customers of a leading AI startup
- These are real deployments — top-down contracts, converted pilots, and active usage — not just signed deals
- This penetration happened faster than any prior enterprise technology wave; large enterprises are betting on newer AI products earlier than they ever have before

**Top use cases by adoption:**
1. **Coding** — Dominant by an order of magnitude. 10-20x productivity gains for engineers. Tight human-in-the-loop. Verifiable output. Tools: Cursor, Claude Code, Codex.
2. **Support** — High volume, well-defined tasks, clear SOPs, verifiable resolution. Easy to A/B test ROI. Low change management (often outsourced to BPOs already).
3. **Search** — Internal knowledge retrieval, industry-specific search. Glean (enterprise), Harvey (legal search), OpenEvidence (medical search).

**Top industries:**
- **Technology** — Always-early adopter, spawned the wave
- **Legal** — Surprising first mover. AI is excellent at parsing dense text, reasoning over contracts, summarizing. Harvey: ~$200M ARR in 3 years.
- **Healthcare** — Medical scribing (Abridge, Ambience), medical search (OpenEvidence), admin automation (Tennr)

**Why these sectors:** Text-based work, rote and repetitive tasks, natural human-in-the-loop oversight, limited regulation, verifiable outputs. Industries requiring physical world interaction, interpersonal relationships, or heavy regulation are lagging.

## Harvey AI: The Vertical AI Playbook

Harvey ($0 → $200M+ ARR in 36 months; ~$11B valuation) is the canonical vertical AI success story. Founders: Winston Weinberg (securities litigator) + Gabriel Pereyra (DeepMind researcher). Seed investor: OpenAI (led $5M round in 2022).

**Why legal AI works:**
- Billable-hour model means there's already a human-in-the-loop review system — minimum viable AI quality is lower
- Legal work is structured: ~10 categories (drafting, document comparison, case law research, etc.)
- Process data exists only inside law firms — no public training data, so proprietary data is a genuine moat
- Prestige signals trust in professional services — top firm adoption creates downstream credibility

**Harvey's growth levers:**

1. **Contrarian customer choice** — Went to Allen & Overy (now A&O Shearman, 4,000 lawyers) when they had 4 employees. If elite firms trust you, everyone downstream trusts you. On the product side: hard problems build defensible systems.

2. **Hyper-personalized demos** — Before every pitch, researched the specific partner, found their public case filings, had Harvey analyze their own work. "Upload their own argument. Tell me how to argue against it. Lawyers are argumentative. Let them fight with the model."

3. **External trust flywheel** — Law firms bring Harvey to their clients. PE fund's law firm builds a workflow → PE fund wants it → PE fund pulls their other law firms. First 50 enterprise customers were all referrals.

4. **Expand then collapse** — First built specialized systems for each legal task type (separate system for drafting vs. due diligence vs. case research). Then unified into single interface where the AI routes users via "nudges." Classic platform move.

5. **GRR focus** — "A lot of investors have been basically just looking at net new ARR... that's a huge mistake." Median seat count doubles within 12 months of deployment.

6. **Domain experts, not just engineers** — ~20% of 460 employees are lawyers. Domain hires: closing enterprise deals (ex-Wachtell partner calling = meeting gets taken), and product design (lawyers define 15 sub-tasks in a workflow, what "good" looks like).

7. **Selling work, not just software** — Revenue-share model where Harvey builds custom workflows with law firms, law firms sell to clients. Budget comes from professional services spend (billions) not tech budget (millions).

**Competitive landscape:** Harvey vs. Legora. Harvey: ~$200M ARR, backed by Sequoia/a16z/Kleiner. Legora: ~$100M ARR est., backed by Benchmark/Bessemer. Both using Claude heavily.

## Harvey's Tech Architecture

- Runs ~6 foundation models (OpenAI, Anthropic, Google, Mistral) with an orchestration layer that picks the best per task
- Proprietary process data from inside law firms sits on top
- 40% of engineering/product/design team is senior infrastructure engineers (moat is the system, not the prompt)

## Harvey's Vertical Model U-Turn (2025-2026)

Harvey initially built a custom-trained case law model in partnership with OpenAI. Lawyers preferred it over GPT-4 97% of the time. The model drove rapid growth: $190M ARR by January 2026, $11B valuation by March 2026, majority of AmLaw 100 as clients.

Then Harvey scrapped the model.

Frontier reasoning models from Google, xAI, OpenAI, and Anthropic started outperforming Harvey's custom legal model on Harvey's own BigLaw Bench evaluation. Harvey now routes tasks across Claude, Gemini, and GPT via a Model Selector.

*The lesson on vertical model durability:* Fine-tuning wins decisively only when:
- Query patterns are genuinely specialized and underrepresented in general training data
- Consequence of errors is high enough to justify sustained investment
- Company has enough distribution to generate meaningful proprietary feedback

For many categories, the better bet remains exceptional workflow infrastructure, skill files, and agentic orchestration on top of frontier models — a fine-tuned model that requires sustained investment to stay ahead of a constantly improving baseline may not hold its advantage.

*Counterexample:* Intercom's fin-cx-retrieval (custom retrieval model for customer service) works because customer service reasoning is structurally different from general language tasks, and 40M+ resolved conversations have compounded the advantage.

Cursor launched Composer 2 (Apr 2026) — a proprietary coding model built on Moonshot AI's Kimi K2.5 with their own continued pre-training and RL. Scored 61.7% on Terminal-Bench 2.0, beating Claude Opus 4.6 (58.0%), at $0.50/M input tokens (1/10th of Anthropic's flagship). Cursor's strategy: frontier models for hardest reasoning tasks, custom vertical models for everything else.

## The 1-1-1 Playbook (Alton Syn / Mark Cuban)

Alton Syn distills Mark Cuban's AI agent thesis into a sharper framework: the money isn't in "selling AI" — it's in pointing to **one painful workflow inside one business** and fixing it fast.

**The framework:** One vertical. One workflow. One painful problem. Everything else is distribution.

**Why most people fail:** They pick a vertical too broad ("healthcare," "real estate," "legal"). The winning move is one level deeper:
- PT clinics with insurance verification delays
- Cleaning companies with slow quote turnaround
- HVAC businesses with missed maintenance follow-up
- Pool service companies with technician routing chaos

**What clients actually buy:** Not agents, not AI, not your stack — they buy the disappearance of a painful operational loop. A strong offer makes sense to a tired business owner in under 15 seconds: "when a job is cancelled, the next best customer gets contacted automatically."

**The real moat — workflow fluency:** Knowing one painful loop well enough to instantly answer: What triggers it? What data does it need? What counts as urgent? Where does a human still step in? What breaks most often? What ROI shows up fastest?

**Where value is shifting:** Away from node-dragging, repetitive setup, and manual debugging as a billing model. Toward workflow selection, commercial packaging, deployment confidence, monitoring, and iteration speed.

**The playbook from zero:**
1. Pick one narrow vertical slice (not "healthcare" — a specific operator pain)
2. Find one workflow tied to money or time (lead response, quote turnaround, cancellations, invoice recovery)
3. Make the offer outcome-based — say what painful thing stops happening
4. Keep V1 tight — one workflow, one commercial win, one before-and-after
5. Build for ownership: monitoring, fallbacks, alerts, exception handling — this is where retainers come from

This aligns with the [Services-as-Software](services-as-software.md) thesis: the 1-1-1 playbook is effectively the individual operator's version of the autopilot wedge — start with the outsourced, intelligence-heavy task and compound from there.

## Comparable Cases

- **Healthcare:** Abridge (medical scribing), Ambience Healthcare, OpenEvidence (medical search), Tennr (back-office healthcare admin). All grew rapidly on discrete, text-heavy use cases that circumvent the EHR system of record. See also [AI Drug Discovery](../science/ai-drug-discovery.md) for how generative AI is compressing preclinical timelines in pharma.
- **Code:** Cursor (reported explosive growth), Claude Code, Codex. Code is "upstream of all other applications" — AI accelerating code accelerates every domain.

## Implications for Builders

- Fertile ground: serving tech, legal, healthcare buyers — but no single winner; many sub-specialties within each
- Look for high model capability but no breakout company yet (the company that builds early when capabilities arrive + has market awareness when they mature wins)
- Watch where labs focus research: long-horizon agents, computer use, spreadsheet/presentation interfaces signal next unlocks

See also: [Business Moats in AI](../concepts/business-moats-in-ai.md), [AI Startup Distribution](ai-startup-distribution.md)

## Sources

- "Harvey AI went from $0 to $200M+ ARR in 36 months" — Ivan Landabaso (tweet thread, Apr 2026) ([link](https://x.com/ivanlandabaso/status/2042179119325082087/?s=12&rw_tt_thread=True))
- "AI Adoption by the Numbers" — Kimberly Tan (a16z, Apr 2026) ([link](https://www.a16z.news/p/ai-adoption-by-the-numbers?r=1xuh9&utm_medium=ios&triedRedirect=true))
- "The New Software: CLI, Skills & Vertical Models" — Sandhya (tweet thread, Apr 2026) ([link](https://x.com))
- "Mark Cuban Is Right About AI Agents" — Alton Syn (tweet, Apr 2026) ([link](https://x.com/altonsyn))
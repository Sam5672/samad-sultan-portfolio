# Samad Sultan — Project Portfolio

Computer Science (BS) student at the **University of Virginia**, pursuing the AI concentration and an Applied Mathematics minor. I build backend systems and machine-learning models, and I care about testing whether they actually work.

## Projects

| Project | What it does | Stack | My role |
|---|---|---|---|
| [**FedSignal**](./fedsignal) | Scores central bank statements for hawkish or dovish tone with a fine-tuned language model, then estimates where short-term bond yields go next. Covers the US, UK, Australia and Pakistan. | Python, PyTorch, Hugging Face (DistilBERT), scikit-learn, FastAPI | Backend algorithm: sentiment model and yield regression (HooHacks 2026, team) |
| [**Solar Installation Time Predictor**](./solar-install-predictor) | Predicts solar installation job length for a real client, Midwest Solar LLC, so crews can be scheduled without running over. | R, randomForest, caret | Built the prediction model (ENGR 1020, team) |
| [**AI Prompt Improver**](./prompt-improver) | Browser extension that rewrites AI chat prompts before sending, grounded only in the draft and visible thread so it never invents content. In progress — extension scaffolding and backend plumbing are done, the real rewrite logic isn't wired up yet. | Manifest V3, JavaScript, Node.js, Express | Solo project — architecture, extension, and backend |

Each project folder has its own README covering the problem, how it works, how to run it, and its limitations along with what I'd improve.

## Experience

- **Software Engineering Intern, CuraQ Health** (Aug–Nov 2026): backend features for the Health Hub platform. I built a FastAPI retrieval microservice on PostgreSQL/pgvector that grounds AI-generated content in cited sources, a pluggable LLM-provider layer, and a compliance-logging API in Laravel, all with automated test suites.
- **Software Engineering Intern, CONTECH International** (Jun–Aug 2023): refactored 20 legacy Java classes to split a hospital management system into product-specific modules, and wrote tests that caught 13 bugs before release.

*Code from these internships belongs to the companies and isn't included here.*

## Skills

**Languages:** Python, Java, R, C, JavaScript, PHP, SQL
**ML / data:** PyTorch, Hugging Face Transformers, scikit-learn, pandas, Random Forests
**Backend:** FastAPI, Laravel, Django, Node.js / Express, REST APIs, PostgreSQL / pgvector, Docker Compose
**Browser extensions:** Manifest V3, content scripts, Chrome extension messaging
**Testing:** PHPUnit, unit and integration testing

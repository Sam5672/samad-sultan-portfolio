import "dotenv/config";
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// Stub: no LLM call yet. Just echoes the draft back so the extension has a
// real response contract to build against before the API key is wired up.
app.post("/rewrite", (req, res) => {
  const { draft } = req.body ?? {};

  if (typeof draft !== "string" || draft.trim() === "") {
    return res.status(400).json({ error: "draft is required" });
  }

  res.json({
    rewritten_prompt: draft,
    changes: [],
    note: "stub response — LLM integration not implemented yet",
  });
});

app.listen(PORT, () => {
  console.log(`AI Prompt Improver backend listening on http://localhost:${PORT}`);
});
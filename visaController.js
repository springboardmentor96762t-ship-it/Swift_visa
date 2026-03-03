import { askLLM } from "../services/llmService.js";

export async function askModel(req, res) {
  const answer = await askLLM(req.body.query);
  res.json({ response: answer });
}

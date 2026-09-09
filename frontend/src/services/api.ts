import type {
  AgentInfo,
  GameResponse,
  HealthResponse,
  PerformanceRow,
  RootAnalysisRow,
  SimulationResult,
  StatsResponse,
  TreeIndexRow,
  TreeNodeResponse,
  TreeSummaryResponse,
  VocabularySummaryRow
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {})
    },
    ...options
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function fetchVocabularySummary(): Promise<VocabularySummaryRow[]> {
  return request<VocabularySummaryRow[]>("/vocabulary/summary");
}

export function fetchTrees(): Promise<{ trees: TreeIndexRow[] }> {
  return request<{ trees: TreeIndexRow[] }>("/trees");
}

export function fetchTreeSummary(length: number, weighting = "uniform"): Promise<TreeSummaryResponse> {
  return request<TreeSummaryResponse>(`/trees/${length}?weighting=${encodeURIComponent(weighting)}`);
}

export function fetchTreeNode(length: number, nodeId: number, weighting = "uniform"): Promise<TreeNodeResponse> {
  return request<TreeNodeResponse>(`/trees/${length}/node/${nodeId}?weighting=${encodeURIComponent(weighting)}`);
}

export function expandTreeNode(
  length: number,
  nodeId: number,
  weighting = "uniform",
  nodeBudget = 20000
): Promise<TreeNodeResponse> {
  return request<TreeNodeResponse>(`/trees/${length}/node/${nodeId}/expand?weighting=${encodeURIComponent(weighting)}`, {
    method: "POST",
    body: JSON.stringify({ node_budget: nodeBudget })
  });
}

export function fetchRootAnalysis(length: number, weighting = "uniform"): Promise<RootAnalysisRow[]> {
  return request<RootAnalysisRow[]>(`/trees/${length}/root-analysis?weighting=${encodeURIComponent(weighting)}`);
}

export function fetchResultsByLength(): Promise<PerformanceRow[]> {
  return request<PerformanceRow[]>("/results/by-length");
}

export function createGame(
  word?: string,
  maxLives = 6,
  length?: number,
  weighting = "uniform"
): Promise<GameResponse> {
  return request<GameResponse>("/game/new", {
    method: "POST",
    body: JSON.stringify({ word: word || null, length: word ? null : length, max_lives: maxLives, weighting })
  });
}

export function guessLetter(gameId: string, letter: string): Promise<GameResponse> {
  return request<GameResponse>(`/game/${gameId}/guess`, {
    method: "POST",
    body: JSON.stringify({ letter })
  });
}

export function treeStep(gameId: string): Promise<GameResponse> {
  return request<GameResponse>(`/game/${gameId}/tree-step`, { method: "POST" });
}

export function treePlay(gameId: string, maxSteps = 26): Promise<GameResponse> {
  return request<GameResponse>(`/game/${gameId}/tree-play`, {
    method: "POST",
    body: JSON.stringify({ max_steps: maxSteps })
  });
}

export function aiGuess(gameId: string, agent = "hangman_decision_tree"): Promise<GameResponse> {
  return request<GameResponse>(`/game/${gameId}/ai-guess`, {
    method: "POST",
    body: JSON.stringify({ agent })
  });
}

export function fetchAgents(): Promise<{ agents: AgentInfo[] }> {
  return request<{ agents: AgentInfo[] }>("/agents");
}

export function fetchStats(): Promise<StatsResponse> {
  return request<StatsResponse>("/stats");
}

export function simulate(agent: string, word: string, games = 1): Promise<SimulationResult> {
  return request<SimulationResult>("/simulate", {
    method: "POST",
    body: JSON.stringify({ agent, word: word || null, games })
  });
}

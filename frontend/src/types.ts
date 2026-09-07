export type GameStatus = "playing" | "won" | "lost";

export type GuessRecord = {
  letter: string;
  correct: boolean;
  positions: number[];
  pattern: string;
  remaining_lives: number;
};

export type GameState = {
  word_length: number;
  pattern: string;
  pattern_display: string;
  guessed_letters: string[];
  correct_letters: string[];
  incorrect_letters: string[];
  remaining_lives: number;
  max_lives: number;
  status: GameStatus;
  turn: number;
  history: GuessRecord[];
};

export type VocabularySummaryRow = {
  length: number;
  number_of_words: number;
  mean_zipf_frequency?: number;
  max_zipf_frequency?: number;
  min_zipf_frequency?: number;
};

export type HealthResponse = {
  status: string;
  vocabulary_size: number;
  available_vocabulary_lengths: number[];
  trained_tree_count: number;
  available_tree_lengths: number[];
  wordfreq_vocabulary_loaded: boolean;
};

export type TreeBranch = {
  outcome_pattern: string;
  node_id: number | null;
  word_count: number;
  probability: number;
  is_miss: boolean;
  resulting_pattern: string;
  remaining_lives: number;
};

export type TreeNodeResponse = {
  node_id: number;
  depth: number;
  pattern: string;
  guessed_letters: string[];
  incorrect_letters: string[];
  remaining_lives: number;
  candidate_count: number;
  guess: string | null;
  leaf_type: string | null;
  children: Record<string, number>;
  branches: TreeBranch[];
  win_probability: number;
  average_mistakes: number;
  average_total_guesses: number;
  average_remaining_lives: number;
  expected_depth: number;
  candidate_sample: string[];
  fallback_letters: string[];
  state_key: string;
};

export type TreeIndexRow = {
  length: number;
  weighting: string;
  model_path: string;
  model_file_size: number;
  best_first_guess?: string;
  node_count?: number;
  leaf_count?: number;
  max_depth?: number;
  win_rate?: number;
  average_mistakes?: number;
  training_strategy?: string;
  training_time?: number;
  words?: number;
};

export type TreeSummaryResponse = TreeIndexRow & {
  root: TreeNodeResponse;
  metadata?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
  training_statistics?: Record<string, unknown>;
  vocabulary?: Record<string, unknown>;
};

export type TreeDecision = {
  tree_length: number;
  weighting: string;
  model_source: string;
  node_id: number;
  next_node_id: number | null;
  branch_found: boolean;
  guess: string;
  outcome_pattern: string;
  resulting_pattern: string;
  node: {
    candidate_count: number;
    win_probability: number;
    average_mistakes: number;
    depth: number;
    branch_count: number;
    fallback_letters: string[];
  };
  branch: Partial<TreeBranch>;
};

export type TreeModelInfo = {
  length: number;
  weighting: string;
  model_source: string;
  current_node_id: number;
  in_vocabulary: boolean;
  best_first_guess?: string;
  node_count: number;
  max_depth: number;
  training_strategy: string;
};

export type GameResponse = {
  game_id: string;
  state: GameState;
  solution?: string;
  decision?: TreeDecision;
  decisions?: TreeDecision[];
  new_decisions?: TreeDecision[];
  model?: TreeModelInfo;
  analysis?: {
    candidate_count: number;
    candidate_sample: string[];
  };
};

export type PerformanceRow = {
  length: number;
  total_words: number;
  wins: number;
  losses: number;
  win_rate: number;
  average_wrong_guesses: number;
  median_wrong_guesses: number;
  average_total_guesses: number;
  average_remaining_lives: number;
  maximum_tree_depth: number;
  average_traversal_depth: number;
  node_count: number;
  leaf_count: number;
  model_file_size: number;
  training_time: number;
  evaluation_time: number;
  best_first_guess: string;
  training_strategy: string;
  weighting: string;
};

export type RootAnalysisRow = {
  length: number;
  root_letter: string;
  words: number;
  win_rate: number;
  average_mistakes: number;
  average_total_guesses: number;
  node_count: number;
  maximum_depth: number;
  training_strategy: string;
  weighting: string;
  training_time: number;
};

export type LetterScore = {
  letter: string;
  score: number;
  probability_present: number;
  entropy: number;
  expected_candidates_after_guess: number;
  outcome_count: number;
};

export type AgentDecision = {
  agent: string;
  guess: string;
  candidate_count: number;
  rankings: LetterScore[];
  metadata: Record<string, unknown>;
};

export type AgentInfo = {
  name: string;
  trained: boolean;
  description: string;
};

export type BenchmarkSummary = {
  agent: string;
  games: string;
  wins: string;
  losses: string;
  win_rate: string;
  loss_rate: string;
  win_rate_ci_low: string;
  win_rate_ci_high: string;
  average_guesses: string;
  average_incorrect_guesses: string;
  average_remaining_lives_when_winning: string;
  average_turns: string;
  average_decision_time_ms: string;
  average_candidate_reduction: string;
};

export type StatsResponse = HealthResponse & {
  performance_by_length: PerformanceRow[];
  dictionary_size?: number;
  word_lengths?: number[];
  decision_tree_model_loaded?: boolean;
  benchmark_summary?: BenchmarkSummary[];
};

export type SimulationResult = {
  agent: string;
  games: number;
  win_rate: number;
  results: Array<{
    word: string;
    status: GameStatus;
    turns: number;
    incorrect_guesses: number;
    remaining_lives: number;
    history: GuessRecord[];
    decisions: TreeDecision[];
  }>;
};

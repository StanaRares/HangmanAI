import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { About } from "./components/About";
import { HangmanFigure } from "./components/HangmanFigure";
import { LetterKeyboard } from "./components/LetterKeyboard";
import {
  createGame,
  expandTreeNode,
  fetchHealth,
  fetchResultsByLength,
  fetchRootAnalysis,
  fetchTreeNode,
  fetchTreeSummary,
  fetchTrees,
  fetchVocabularySummary,
  guessLetter,
  treePlay,
  treeStep
} from "./services/api";
import type {
  GameResponse,
  HealthResponse,
  PerformanceRow,
  RootAnalysisRow,
  TreeBranch,
  TreeIndexRow,
  TreeNodeResponse,
  TreeSummaryResponse,
  VocabularySummaryRow
} from "./types";

type Tab = "play" | "tree" | "experiments" | "roots" | "research";

type ExperimentRow = {
  length: number;
  total_words: number;
  best_first_guess: string;
  win_rate: number;
  average_wrong_guesses: number;
  node_count: number;
  maximum_tree_depth: number;
  training_strategy: string;
  training_time: number;
};

const tabs: Array<{ id: Tab; label: string }> = [
  { id: "play", label: "Play" },
  { id: "tree", label: "Tree" },
  { id: "experiments", label: "Experiments" },
  { id: "roots", label: "Roots" },
  { id: "research", label: "Research" }
];

function integer(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "n/a";
}

function fixed(value: number | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function percent(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
}

function singleLetter(value: string | undefined | null) {
  const normalized = value?.trim().toLowerCase();
  return normalized && /^[a-z]$/.test(normalized) ? normalized : undefined;
}

function guessDisplay(value: string | undefined | null, fallback = "fixed") {
  return singleLetter(value)?.toUpperCase() ?? fallback;
}

function isCheckpointNode(node: TreeNodeResponse) {
  return node.guess === null && node.expandable && (node.leaf_type === "budget" || node.leaf_type === "depth_limited");
}

function checkpointReason(node: TreeNodeResponse) {
  return node.leaf_type === "budget"
    ? "This branch reached the construction budget."
    : "This branch reached the configured depth limit.";
}

function terminalLabel(node: TreeNodeResponse) {
  if (node.leaf_type === "deterministic") return "fixed";
  return node.leaf_type ?? "terminal";
}

function patternDisplay(pattern: string | undefined) {
  return pattern ? pattern.split("").join(" ") : "";
}

function nodeLabel(nodeId: number | null | undefined) {
  if (nodeId === null || typeof nodeId === "undefined") return "n/a";
  return String(nodeId).padStart(4, "0");
}

function normalizeExperimentRows(results: PerformanceRow[], trees: TreeIndexRow[], vocabulary: VocabularySummaryRow[]) {
  const vocabByLength = new Map(vocabulary.map((row) => [Number(row.length), Number(row.number_of_words)]));
  if (results.length) {
    return results.map(
      (row): ExperimentRow => ({
        length: Number(row.length),
        total_words: Number(row.total_words),
        best_first_guess: row.best_first_guess,
        win_rate: Number(row.win_rate),
        average_wrong_guesses: Number(row.average_wrong_guesses),
        node_count: Number(row.node_count),
        maximum_tree_depth: Number(row.maximum_tree_depth),
        training_strategy: row.training_strategy,
        training_time: Number(row.training_time)
      })
    );
  }
  return trees.map(
    (row): ExperimentRow => ({
      length: Number(row.length),
      total_words: Number(row.words ?? vocabByLength.get(Number(row.length)) ?? 0),
      best_first_guess: row.best_first_guess ?? "",
      win_rate: Number(row.win_rate ?? 0),
      average_wrong_guesses: Number(row.average_mistakes ?? 0),
      node_count: Number(row.node_count ?? 0),
      maximum_tree_depth: Number(row.max_depth ?? 0),
      training_strategy: row.training_strategy ?? "",
      training_time: Number(row.training_time ?? 0)
    })
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("play");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [vocabulary, setVocabulary] = useState<VocabularySummaryRow[]>([]);
  const [trees, setTrees] = useState<TreeIndexRow[]>([]);
  const [results, setResults] = useState<PerformanceRow[]>([]);
  const [selectedLength, setSelectedLength] = useState(5);
  const [playLength, setPlayLength] = useState(5);
  const [secretWord, setSecretWord] = useState("");
  const [game, setGame] = useState<GameResponse | null>(null);
  const [treeSummary, setTreeSummary] = useState<TreeSummaryResponse | null>(null);
  const [treeNode, setTreeNode] = useState<TreeNodeResponse | null>(null);
  const [nodePath, setNodePath] = useState<number[]>([]);
  const [rootAnalysis, setRootAnalysis] = useState<RootAnalysisRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const trainedLengths = useMemo(
    () => Array.from(new Set(trees.filter((tree) => tree.weighting === "uniform").map((tree) => Number(tree.length)))).sort((a, b) => a - b),
    [trees]
  );
  const vocabularyLengths = useMemo(
    () => Array.from(new Set(vocabulary.map((row) => Number(row.length)))).sort((a, b) => a - b),
    [vocabulary]
  );
  const lengthOptions = trainedLengths.length ? trainedLengths : vocabularyLengths;
  const experimentRows = useMemo(
    () => normalizeExperimentRows(results, trees.filter((tree) => tree.weighting === "uniform"), vocabulary),
    [results, trees, vocabulary]
  );
  const playStats = experimentRows.find((row) => row.length === playLength);
  const selectedStats = experimentRows.find((row) => row.length === selectedLength);
  const hardestLength = experimentRows.reduce<ExperimentRow | null>((hardest, row) => {
    if (!hardest) return row;
    return row.win_rate < hardest.win_rate ? row : hardest;
  }, null);
  const easiestLength = experimentRows.reduce<ExperimentRow | null>((easiest, row) => {
    if (!easiest) return row;
    return row.win_rate > easiest.win_rate ? row : easiest;
  }, null);
  const chartRows = experimentRows.map((row) => ({
    ...row,
    label: String(row.length).padStart(2, "0"),
    winRatePct: row.win_rate * 100
  }));
  const rootChartRows = rootAnalysis.map((row) => ({
    letter: row.root_letter.toUpperCase(),
    winRatePct: Number(row.win_rate) * 100,
    mistakes: Number(row.average_mistakes)
  }));

  useEffect(() => {
    void refreshData();
  }, []);

  useEffect(() => {
    if (!lengthOptions.length) return;
    if (!lengthOptions.includes(selectedLength)) {
      setSelectedLength(lengthOptions[0]);
    }
    if (!lengthOptions.includes(playLength)) {
      setPlayLength(lengthOptions[0]);
    }
  }, [lengthOptions, playLength, selectedLength]);

  useEffect(() => {
    if (!selectedLength || !trainedLengths.includes(selectedLength)) {
      setTreeSummary(null);
      setTreeNode(null);
      setRootAnalysis([]);
      return;
    }
    let cancelled = false;
    setError(null);
    Promise.all([fetchTreeSummary(selectedLength), fetchRootAnalysis(selectedLength)])
      .then(([summary, roots]) => {
        if (cancelled) return;
        setTreeSummary(summary);
        setTreeNode(summary.root);
        setNodePath([summary.root.node_id]);
        setRootAnalysis(roots);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load tree");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedLength, trainedLengths]);

  async function refreshData() {
    setLoading(true);
    setError(null);
    try {
      const [nextHealth, treeResponse, vocabRows, resultRows] = await Promise.all([
        fetchHealth(),
        fetchTrees(),
        fetchVocabularySummary(),
        fetchResultsByLength()
      ]);
      setHealth(nextHealth);
      setTrees(treeResponse.trees);
      setVocabulary(vocabRows);
      setResults(resultRows);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backend unavailable");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function startGame() {
    setLoading(true);
    setError(null);
    try {
      const word = secretWord.trim().toLowerCase();
      const response = await createGame(word || undefined, 6, playLength);
      setGame(response);
      setActiveTab("play");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start game");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function handleManualGuess(letter: string) {
    if (!game) return;
    setLoading(true);
    setError(null);
    try {
      setGame(await guessLetter(game.game_id, letter));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Guess failed");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function handleTreeStep() {
    if (!game) return;
    setLoading(true);
    setError(null);
    try {
      setGame(await treeStep(game.game_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tree step failed");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function handleTreePlay() {
    if (!game) return;
    setLoading(true);
    setError(null);
    try {
      setGame(await treePlay(game.game_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tree play failed");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function loadNode(branch: TreeBranch) {
    if (branch.node_id === null) {
      setError("Tree consistency error: selected branch has no destination node.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage("Materializing subtree...");
      const next = await fetchTreeNode(selectedLength, branch.node_id, "uniform", true);
      setTreeNode(next);
      setNodePath((current) => [...current, next.node_id]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load node");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  async function loadRootNode() {
    if (!treeSummary) return;
    setTreeNode(treeSummary.root);
    setNodePath([treeSummary.root.node_id]);
  }

  async function expandCurrentNode() {
    if (!treeNode?.expandable) return;
    setLoading(true);
    setError(null);
    try {
      setLoadingMessage("Materializing subtree...");
      const expanded = await expandTreeNode(selectedLength, treeNode.node_id);
      setTreeNode(expanded);
      setNodePath((current) => (current[current.length - 1] === expanded.node_id ? current : [...current, expanded.node_id]));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not expand branch");
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  }

  const decisionHistory = game?.decisions ?? [];
  const currentDecision = game?.decision ?? decisionHistory[decisionHistory.length - 1];
  const featuredGuess = currentDecision?.guess ?? singleLetter(playStats?.best_first_guess);
  const pathText = decisionHistory.length
    ? decisionHistory
        .map((decision) => `${nodeLabel(decision.node_id)} -${decision.guess.toUpperCase()}/${decision.outcome_pattern}-> ${nodeLabel(decision.next_node_id)}`)
        .join("  ")
    : "0000";

  return (
    <main className="app-shell">
      <header className="site-header">
        <button className="wordmark" type="button" onClick={() => setActiveTab("play")}>
          HangmanAI
        </button>
        <nav className="site-nav" aria-label="Primary">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={activeTab === tab.id ? "active" : ""}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {error && <div className="error global-error">{error}</div>}
      {loadingMessage && <div className="loading-banner">{loadingMessage}</div>}

      {activeTab === "play" && (
        <section className="play-page">
          <div className="play-kicker">
            <span>{playLength} letter tree</span>
            <span>{integer(playStats?.total_words)} words</span>
            <span>{playStats?.training_strategy ?? "tree"}</span>
          </div>
          <div className="play-question">
            <p>What is the best Hangman strategy</p>
            <h1>for a {playLength}-letter word?</h1>
          </div>

          <div className="play-stage">
            <section className="game-theater" aria-label="Hangman game">
              <HangmanFigure state={game?.state} />
              <div className="pattern-line">{game?.state.pattern_display ?? patternDisplay("_".repeat(playLength))}</div>
              <div className="ai-guess">
                <span>Tree guesses</span>
                <strong>{guessDisplay(featuredGuess, "?")}</strong>
              </div>
              <LetterKeyboard
                state={game?.state}
                onGuess={handleManualGuess}
                disabled={loading || !game}
                highlightedLetter={featuredGuess}
              />
              <div className="game-status-line">
                <span>{game?.state.remaining_lives ?? 6} lives</span>
                <span>{game?.state.incorrect_letters.join(" ").toUpperCase() || "no misses"}</span>
                <span className={`status ${game?.state.status ?? "ready"}`}>{game?.state.status ?? "ready"}</span>
                {game?.solution && <span>solution {game.solution}</span>}
              </div>
            </section>

            <aside className="experiment-rail" aria-label="Experiment controls and tree state">
              <section className="control-strip">
                <label>
                  <span>Hidden word</span>
                  <input
                    value={secretWord}
                    onChange={(event) => setSecretWord(event.target.value.toLowerCase())}
                    placeholder="random from selected length"
                  />
                </label>
                <label>
                  <span>Length</span>
                  <select value={playLength} onChange={(event) => setPlayLength(Number(event.target.value))}>
                    {lengthOptions.map((length) => (
                      <option key={length} value={length}>
                        {length} letters
                      </option>
                    ))}
                  </select>
                </label>
                <div className="action-line">
                  <button type="button" onClick={() => void startGame()} disabled={loading}>
                    New word
                  </button>
                  <button type="button" onClick={() => void handleTreeStep()} disabled={loading || !game || game.state.status !== "playing"}>
                    Step tree
                  </button>
                  <button type="button" className="run-action" onClick={() => void handleTreePlay()} disabled={loading || !game || game.state.status !== "playing"}>
                    Run tree
                  </button>
                </div>
              </section>

              <section className="info-list">
                <h2>Inside the tree</h2>
                <dl>
                  <dt>Tree position</dt>
                  <dd>{nodeLabel(game?.model?.current_node_id)}</dd>
                  <dt>Candidates</dt>
                  <dd>{integer(game?.analysis?.candidate_count)}</dd>
                  <dt>Depth</dt>
                  <dd>{integer(currentDecision?.node.depth)}</dd>
                  <dt>Strategy</dt>
                  <dd>{game?.model?.training_strategy ?? playStats?.training_strategy ?? "unloaded"}</dd>
                </dl>
              </section>

              <section className="why-block">
                <h2>{currentDecision ? `Why ${currentDecision.guess.toUpperCase()}?` : "Why this letter?"}</h2>
                <p>
                  {currentDecision
                    ? `Node ${nodeLabel(currentDecision.node_id)} selects ${currentDecision.guess.toUpperCase()}; the observed branch ${currentDecision.outcome_pattern} leads to node ${nodeLabel(currentDecision.next_node_id)}.`
                    : `The opening node for this length currently favors ${guessDisplay(playStats?.best_first_guess, "a fixed branch")} under the recorded ${playStats?.training_strategy ?? "tree"} policy.`}
                </p>
              </section>

              <section className="tree-path">
                <h2>Tree path</h2>
                <code>{pathText}</code>
              </section>
            </aside>
          </div>
        </section>
      )}

      {activeTab === "tree" && (
        <section className="tree-page">
          <header className="editorial-heading">
            <p>Lazy tree inspection</p>
            <h1>One node at a time.</h1>
          </header>
          <div className="length-selector">
            <label>
              Length
              <select value={selectedLength} onChange={(event) => setSelectedLength(Number(event.target.value))}>
                {lengthOptions.map((length) => (
                  <option key={length} value={length}>
                    {length} letters
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => void loadRootNode()}>
              Root node
            </button>
          </div>

          {treeNode ? (
            <div className="tree-composition">
              <section className={`node-sheet ${isCheckpointNode(treeNode) ? "checkpoint-node" : ""}`}>
                <span>Node {nodeLabel(treeNode.node_id)}</span>
                {treeNode.guess ? (
                  <>
                    <p>Guess</p>
                    <strong>{treeNode.guess.toUpperCase()}</strong>
                    <code>{patternDisplay(treeNode.pattern)}</code>
                  </>
                ) : isCheckpointNode(treeNode) ? (
                  <div className="checkpoint-copy">
                    <p>Tree checkpoint</p>
                    <h2>This subtree could not be materialized.</h2>
                    <p>{checkpointReason(treeNode)}</p>
                    <dl>
                      <dt>Pattern</dt>
                      <dd>
                        <code>{patternDisplay(treeNode.pattern)}</code>
                      </dd>
                      <dt>Candidates</dt>
                      <dd>{integer(treeNode.reconstructed_candidate_count)}</dd>
                      <dt>Lives</dt>
                      <dd>{treeNode.remaining_lives}</dd>
                    </dl>
                    <button type="button" onClick={() => void expandCurrentNode()} disabled={loading}>
                      Retry materialization
                    </button>
                  </div>
                ) : (
                  <>
                    <p>Terminal outcome</p>
                    <strong>{terminalLabel(treeNode)}</strong>
                    <code>{patternDisplay(treeNode.pattern)}</code>
                  </>
                )}
              </section>

              <section className="info-list node-stats">
                <h2>State estimate</h2>
                <dl>
                  <dt>Candidates</dt>
                  <dd>{integer(treeNode.candidate_count)}</dd>
                  <dt>Lives</dt>
                  <dd>{treeNode.remaining_lives}</dd>
                  <dt>Win estimate</dt>
                  <dd>{percent(treeNode.win_probability)}</dd>
                  <dt>Avg mistakes</dt>
                  <dd>{fixed(treeNode.average_mistakes)}</dd>
                  <dt>Branches</dt>
                  <dd>{treeNode.branches.length}</dd>
                  <dt>State key</dt>
                  <dd>{treeNode.state_key.slice(0, 12)}</dd>
                </dl>
              </section>

              <section className="branch-ledger">
                <h2>Branches</h2>
                <div className="path-strip">{nodePath.map((node) => nodeLabel(node)).join(" / ")}</div>
                <div className="branch-list">
                  {treeNode.branches.length ? (
                    treeNode.branches.map((branch) => (
                      <button
                        key={`${branch.outcome_pattern}-${branch.node_id}`}
                        type="button"
                        onClick={() => void loadNode(branch)}
                        disabled={branch.node_id === null || loading}
                      >
                        <code>{branch.outcome_pattern}</code>
                        <span>{integer(branch.word_count)} words</span>
                        <span>{patternDisplay(branch.resulting_pattern)}</span>
                      </button>
                    ))
                  ) : (
                    <p className="empty-state">
                      {treeNode.expandable
                        ? "This subtree could not be materialized. Retry from the checkpoint panel."
                        : "This terminal node has no further outcome branches."}
                    </p>
                  )}
                </div>
              </section>

              <section className="word-samples">
                <h2>Words reaching this node</h2>
                <div>
                  {treeNode.candidate_sample.map((candidate) => (
                    <span key={candidate}>{candidate}</span>
                  ))}
                </div>
              </section>
            </div>
          ) : (
            <p className="empty-state">No serialized tree is available for this length yet.</p>
          )}
        </section>
      )}

      {activeTab === "experiments" && (
        <section className="experiments-page">
          <header className="editorial-heading wide-heading">
            <p>Experiment results</p>
            <h1>Short words are the hard case.</h1>
          </header>

          <section className="result-sweep">
            <div>
              <span>Total vocabulary</span>
              <strong>{integer(health?.vocabulary_size)}</strong>
            </div>
            <div>
              <span>Trained trees</span>
              <strong>{health?.trained_tree_count ?? 0}</strong>
            </div>
            <div>
              <span>Hardest length</span>
              <strong>{hardestLength ? String(hardestLength.length).padStart(2, "0") : "n/a"}</strong>
            </div>
            <div>
              <span>Easiest length</span>
              <strong>{easiestLength ? String(easiestLength.length).padStart(2, "0") : "n/a"}</strong>
            </div>
          </section>

          <section className="opening-strip">
            <h2>Best opening guess</h2>
            <div>
              {experimentRows.map((row) => (
                <span key={row.length}>
                  <code>{String(row.length).padStart(2, "0")}</code>
                  <strong>{guessDisplay(row.best_first_guess, "fixed")}</strong>
                </span>
              ))}
            </div>
          </section>

          <div className="chart-spread">
            <ChartBlock title="Win rate by length" rows={chartRows} dataKey="winRatePct" unit="%" color="#D7FF64" chart="line" />
            <ChartBlock title="Average mistakes" rows={chartRows} dataKey="average_wrong_guesses" color="#E0A96D" chart="line" />
            <ChartBlock title="Tree size" rows={chartRows} dataKey="node_count" color="#9CB8B3" chart="bar" />
            <ChartBlock title="Vocabulary size" rows={chartRows} dataKey="total_words" color="#C8A7D8" chart="bar" />
          </div>

          <section className="results-table">
            <table>
              <thead>
                <tr>
                  <th>Length</th>
                  <th>Words</th>
                  <th>First</th>
                  <th>Win</th>
                  <th>Mistakes</th>
                  <th>Nodes</th>
                  <th>Depth</th>
                  <th>Strategy</th>
                </tr>
              </thead>
              <tbody>
                {experimentRows.map((row) => (
                  <tr key={row.length}>
                    <td>{String(row.length).padStart(2, "0")}</td>
                    <td>{integer(row.total_words)}</td>
                    <td>{guessDisplay(row.best_first_guess, "fixed")}</td>
                    <td>{percent(row.win_rate)}</td>
                    <td>{fixed(row.average_wrong_guesses)}</td>
                    <td>{integer(row.node_count)}</td>
                    <td>{integer(row.maximum_tree_depth)}</td>
                    <td>{row.training_strategy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </section>
      )}

      {activeTab === "roots" && (
        <section className="roots-page">
          <header className="editorial-heading">
            <p>Root letter analysis</p>
            <h1>Which letter should you guess first?</h1>
          </header>
          <div className="root-intro">
            <label>
              Word length
              <select value={selectedLength} onChange={(event) => setSelectedLength(Number(event.target.value))}>
                {lengthOptions.map((length) => (
                  <option key={length} value={length}>
                    {length} letters
                  </option>
                ))}
              </select>
            </label>
            <p>
              {selectedStats
                ? `${integer(selectedStats.total_words)} candidates. The trained tree opens with ${guessDisplay(selectedStats.best_first_guess, "a fixed branch")}.`
                : "Choose a trained length to inspect its forced-root sweep."}
            </p>
          </div>

          {rootChartRows.length ? (
            <section className="root-analysis">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={rootChartRows}>
                  <CartesianGrid strokeDasharray="1 6" stroke="#292C2D" vertical={false} />
                  <XAxis dataKey="letter" stroke="#8C918F" tickLine={false} axisLine={false} />
                  <YAxis stroke="#8C918F" unit="%" tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "#151718", border: "1px solid #292C2D", color: "#F1EEE8" }} />
                  <Bar dataKey="winRatePct" fill="#D7FF64" />
                </BarChart>
              </ResponsiveContainer>

              <div className="root-list">
                {rootAnalysis.map((row, index) => (
                  <div key={row.root_letter} className="root-row">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{row.root_letter.toUpperCase()}</strong>
                    <div className="hairline-bar">
                      <span style={{ width: `${Math.max(2, Number(row.win_rate) * 100)}%` }} />
                    </div>
                    <small>
                      {percent(Number(row.win_rate))} / {fixed(Number(row.average_mistakes))} mistakes
                    </small>
                  </div>
                ))}
              </div>
            </section>
          ) : (
            <p className="empty-state">No root-letter analysis has been generated for this length.</p>
          )}
        </section>
      )}

      {activeTab === "research" && (
        <About
          vocabularySize={health?.vocabulary_size}
          trainedTrees={health?.trained_tree_count}
          hardestLength={hardestLength}
          easiestLength={easiestLength}
        />
      )}
    </main>
  );
}

function ChartBlock({
  title,
  rows,
  dataKey,
  unit,
  color,
  chart
}: {
  title: string;
  rows: Array<Record<string, number | string>>;
  dataKey: string;
  unit?: string;
  color: string;
  chart: "line" | "bar";
}) {
  return (
    <section className="chart-block">
      <h2>{title}</h2>
      {rows.length ? (
        <ResponsiveContainer width="100%" height={260}>
          {chart === "line" ? (
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 6 }}>
              <CartesianGrid strokeDasharray="1 6" stroke="#292C2D" vertical={false} />
              <XAxis dataKey="label" stroke="#8C918F" tickLine={false} axisLine={false} />
              <YAxis stroke="#8C918F" unit={unit} tickLine={false} axisLine={false} width={58} />
              <Tooltip contentStyle={{ background: "#151718", border: "1px solid #292C2D", color: "#F1EEE8" }} />
              <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          ) : (
            <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 6 }}>
              <CartesianGrid strokeDasharray="1 6" stroke="#292C2D" vertical={false} />
              <XAxis dataKey="label" stroke="#8C918F" tickLine={false} axisLine={false} />
              <YAxis stroke="#8C918F" unit={unit} tickLine={false} axisLine={false} width={58} />
              <Tooltip contentStyle={{ background: "#151718", border: "1px solid #292C2D", color: "#F1EEE8" }} />
              <Bar dataKey={dataKey} fill={color} />
            </BarChart>
          )}
        </ResponsiveContainer>
      ) : (
        <p className="empty-state">No rows available.</p>
      )}
    </section>
  );
}

export default App;

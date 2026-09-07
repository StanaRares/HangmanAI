import {
  Activity,
  BarChart3,
  BrainCircuit,
  GitBranch,
  ListTree,
  Play,
  RefreshCw,
  Search,
  Split,
  StepForward
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  createGame,
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
import { About } from "./components/About";
import { HangmanFigure } from "./components/HangmanFigure";
import { LetterKeyboard } from "./components/LetterKeyboard";
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

type Tab = "play" | "tree" | "dashboard" | "roots" | "about";

type DashboardRow = {
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

const tabs: Array<{ id: Tab; label: string; icon: typeof Play }> = [
  { id: "play", label: "Play", icon: Play },
  { id: "tree", label: "Tree Explorer", icon: GitBranch },
  { id: "dashboard", label: "Lengths", icon: BarChart3 },
  { id: "roots", label: "Roots", icon: Split },
  { id: "about", label: "About", icon: Search }
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

function patternDisplay(pattern: string | undefined) {
  return pattern ? pattern.split("").join(" ") : "";
}

function normalizeDashboardRows(results: PerformanceRow[], trees: TreeIndexRow[], vocabulary: VocabularySummaryRow[]) {
  const vocabByLength = new Map(vocabulary.map((row) => [Number(row.length), Number(row.number_of_words)]));
  if (results.length) {
    return results.map(
      (row): DashboardRow => ({
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
    (row): DashboardRow => ({
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
  const dashboardRows = useMemo(
    () => normalizeDashboardRows(results, trees.filter((tree) => tree.weighting === "uniform"), vocabulary),
    [results, trees, vocabulary]
  );
  const chartRows = dashboardRows.map((row) => ({
    ...row,
    label: `${row.length}`,
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
    }
  }

  async function loadNode(branch: TreeBranch) {
    if (branch.node_id === null) return;
    setLoading(true);
    setError(null);
    try {
      const next = await fetchTreeNode(selectedLength, branch.node_id);
      setTreeNode(next);
      setNodePath((current) => [...current, next.node_id]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load node");
    } finally {
      setLoading(false);
    }
  }

  async function loadRootNode() {
    if (!treeSummary) return;
    setTreeNode(treeSummary.root);
    setNodePath([treeSummary.root.node_id]);
  }

  const decisionHistory = game?.decisions ?? [];
  const currentDecision = game?.decision ?? decisionHistory[decisionHistory.length - 1];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Learning the best Hangman tree by word length</p>
          <h1>HangmanAI</h1>
        </div>
        <nav>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                type="button"
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => setActiveTab(tab.id)}
                title={tab.label}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {error && <div className="error global-error">{error}</div>}

      {activeTab === "play" && (
        <section className="play-layout">
          <div className="panel game-panel">
            <div className="section-heading">
              <BrainCircuit size={20} />
              <h2>Decision Tree Player</h2>
            </div>
            <HangmanFigure state={game?.state} />
            <div className="word-display">{game?.state.pattern_display ?? patternDisplay("_".repeat(playLength))}</div>
            <div className="state-strip">
              <span>{game?.state.remaining_lives ?? 6} lives</span>
              <span>{game?.state.incorrect_letters.join(" ").toUpperCase() || "no misses"}</span>
              <span className={`status ${game?.state.status ?? "playing"}`}>{game?.state.status ?? "playing"}</span>
              {game?.model && <span>{game.model.length}-letter tree</span>}
            </div>
            {game?.solution && <div className="solution">Solution: {game.solution}</div>}
            <LetterKeyboard state={game?.state} onGuess={handleManualGuess} disabled={loading || !game} />
          </div>

          <aside className="side-stack">
            <div className="panel controls-panel">
              <div className="section-heading">
                <RefreshCw size={20} />
                <h2>Play Controls</h2>
              </div>
              <label className="field">
                Hidden word
                <input
                  value={secretWord}
                  onChange={(event) => setSecretWord(event.target.value.toLowerCase())}
                  placeholder="random from selected length"
                />
              </label>
              <label className="field">
                Word length
                <select value={playLength} onChange={(event) => setPlayLength(Number(event.target.value))}>
                  {lengthOptions.map((length) => (
                    <option key={length} value={length}>
                      {length} letters
                    </option>
                  ))}
                </select>
              </label>
              <div className="button-row">
                <button type="button" className="secondary-button" onClick={() => void startGame()} disabled={loading}>
                  <RefreshCw size={18} />
                  New
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void handleTreeStep()}
                  disabled={loading || !game || game.state.status !== "playing"}
                  title="Advance one tree decision"
                >
                  <StepForward size={18} />
                  Step
                </button>
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => void handleTreePlay()}
                  disabled={loading || !game || game.state.status !== "playing"}
                  title="Let the tree finish this game"
                >
                  <Play size={18} />
                  Run
                </button>
              </div>
            </div>

            <div className="panel">
              <div className="section-heading">
                <ListTree size={20} />
                <h2>Current Traversal</h2>
              </div>
              <div className="metric-grid">
                <div>
                  <span>node</span>
                  <strong>#{game?.model?.current_node_id ?? 0}</strong>
                </div>
                <div>
                  <span>source</span>
                  <strong>{game?.model?.model_source ?? "none"}</strong>
                </div>
                <div>
                  <span>candidates</span>
                  <strong>{integer(game?.analysis?.candidate_count)}</strong>
                </div>
                <div>
                  <span>strategy</span>
                  <strong>{game?.model?.training_strategy ?? "unloaded"}</strong>
                </div>
              </div>
            </div>

            <div className="panel decision-panel">
              <div className="section-heading">
                <GitBranch size={20} />
                <h2>Last Decision</h2>
              </div>
              {currentDecision ? (
                <div className="decision-card">
                  <div className="decision-letter">{currentDecision.guess.toUpperCase()}</div>
                  <dl>
                    <dt>Tree node</dt>
                    <dd>#{currentDecision.node_id}</dd>
                    <dt>Outcome</dt>
                    <dd>{currentDecision.outcome_pattern}</dd>
                    <dt>Pattern</dt>
                    <dd>{patternDisplay(currentDecision.resulting_pattern)}</dd>
                    <dt>Branch</dt>
                    <dd>{currentDecision.branch_found ? `#${currentDecision.next_node_id}` : "fallback"}</dd>
                  </dl>
                </div>
              ) : (
                <div className="empty-state">No decision has been made in this game.</div>
              )}
            </div>
          </aside>
        </section>
      )}

      {activeTab === "tree" && (
        <section className="tree-layout">
          <div className="panel tree-main">
            <div className="section-heading">
              <GitBranch size={20} />
              <h2>Tree Explorer</h2>
            </div>
            <div className="toolbar">
              <select value={selectedLength} onChange={(event) => setSelectedLength(Number(event.target.value))}>
                {lengthOptions.map((length) => (
                  <option key={length} value={length}>
                    {length} letters
                  </option>
                ))}
              </select>
              <button type="button" className="secondary-button icon-button" onClick={() => void loadRootNode()} title="Root node">
                <ListTree size={18} />
              </button>
            </div>
            {treeNode ? (
              <div className="node-grid">
                <div className="node-spotlight">
                  <span>node #{treeNode.node_id}</span>
                  <strong>{treeNode.guess ? treeNode.guess.toUpperCase() : treeNode.leaf_type}</strong>
                  <small>{patternDisplay(treeNode.pattern)}</small>
                </div>
                <div className="metric-grid">
                  <div>
                    <span>words</span>
                    <strong>{integer(treeNode.candidate_count)}</strong>
                  </div>
                  <div>
                    <span>win</span>
                    <strong>{percent(treeNode.win_probability)}</strong>
                  </div>
                  <div>
                    <span>mistakes</span>
                    <strong>{fixed(treeNode.average_mistakes)}</strong>
                  </div>
                  <div>
                    <span>branches</span>
                    <strong>{treeNode.branches.length}</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">No serialized tree is available for this length yet.</div>
            )}
          </div>

          <div className="panel branch-panel">
            <div className="section-heading">
              <Split size={20} />
              <h2>Branches</h2>
            </div>
            <div className="path-strip">{nodePath.map((node) => `#${node}`).join(" / ")}</div>
            <div className="branch-list">
              {(treeNode?.branches ?? []).map((branch) => (
                <button
                  key={`${branch.outcome_pattern}-${branch.node_id}`}
                  type="button"
                  onClick={() => void loadNode(branch)}
                  disabled={branch.node_id === null || loading}
                  title={`Branch ${branch.outcome_pattern}`}
                >
                  <span>{branch.outcome_pattern}</span>
                  <strong>{patternDisplay(branch.resulting_pattern)}</strong>
                  <small>
                    {integer(branch.word_count)} words · {percent(branch.probability)}
                  </small>
                </button>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="section-heading">
              <Activity size={20} />
              <h2>Samples</h2>
            </div>
            <div className="candidate-list">
              {(treeNode?.candidate_sample ?? []).map((candidate) => (
                <span key={candidate}>{candidate}</span>
              ))}
            </div>
          </div>
        </section>
      )}

      {activeTab === "dashboard" && (
        <section className="dashboard-layout">
          <div className="panel summary-panel">
            <div className="section-heading">
              <BarChart3 size={20} />
              <h2>Word-Length Dashboard</h2>
            </div>
            <div className="metric-grid wide">
              <div>
                <span>usable words</span>
                <strong>{integer(health?.vocabulary_size)}</strong>
              </div>
              <div>
                <span>trained trees</span>
                <strong>{health?.trained_tree_count ?? 0}</strong>
              </div>
              <div>
                <span>length buckets</span>
                <strong>{vocabulary.length}</strong>
              </div>
              <div>
                <span>wordfreq</span>
                <strong>{health?.wordfreq_vocabulary_loaded ? "loaded" : "sample"}</strong>
              </div>
            </div>
            <div className="first-letter-strip">
              {dashboardRows.map((row) => (
                <span key={row.length}>
                  {row.length}: {row.best_first_guess ? row.best_first_guess.toUpperCase() : "?"}
                </span>
              ))}
            </div>
          </div>

          <ChartPanel title="Win Rate" rows={chartRows} dataKey="winRatePct" unit="%" color="#54d6a7" chart="line" />
          <ChartPanel title="Average Mistakes" rows={chartRows} dataKey="average_wrong_guesses" color="#f6b86a" chart="line" />
          <ChartPanel title="Tree Nodes" rows={chartRows} dataKey="node_count" color="#79a7ff" chart="bar" />
          <ChartPanel title="Vocabulary Size" rows={chartRows} dataKey="total_words" color="#e778a6" chart="bar" />

          <div className="panel table-panel">
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
                  <th>Train s</th>
                </tr>
              </thead>
              <tbody>
                {dashboardRows.map((row) => (
                  <tr key={row.length}>
                    <td>{row.length}</td>
                    <td>{integer(row.total_words)}</td>
                    <td>{row.best_first_guess.toUpperCase()}</td>
                    <td>{percent(row.win_rate)}</td>
                    <td>{fixed(row.average_wrong_guesses)}</td>
                    <td>{integer(row.node_count)}</td>
                    <td>{integer(row.maximum_tree_depth)}</td>
                    <td>{row.training_strategy}</td>
                    <td>{fixed(row.training_time, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!dashboardRows.length && <div className="empty-state">No evaluated tree reports are available yet.</div>}
          </div>
        </section>
      )}

      {activeTab === "roots" && (
        <section className="roots-layout">
          <div className="panel root-chart-panel">
            <div className="section-heading">
              <Split size={20} />
              <h2>Root Letter Explorer</h2>
            </div>
            <div className="toolbar">
              <select value={selectedLength} onChange={(event) => setSelectedLength(Number(event.target.value))}>
                {lengthOptions.map((length) => (
                  <option key={length} value={length}>
                    {length} letters
                  </option>
                ))}
              </select>
              <button type="button" className="secondary-button icon-button" onClick={() => void refreshData()} title="Refresh">
                <RefreshCw size={18} />
              </button>
            </div>
            {rootChartRows.length ? (
              <ResponsiveContainer width="100%" height={420}>
                <BarChart data={rootChartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#263542" />
                  <XAxis dataKey="letter" stroke="#9fb0be" />
                  <YAxis stroke="#9fb0be" unit="%" />
                  <Tooltip contentStyle={{ background: "#101821", border: "1px solid #263542" }} />
                  <Bar dataKey="winRatePct" fill="#54d6a7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">No root-letter analysis has been generated for this length.</div>
            )}
          </div>

          <div className="panel root-rank-panel">
            <div className="section-heading">
              <ListTree size={20} />
              <h2>Ranked Roots</h2>
            </div>
            <div className="root-list">
              {rootAnalysis.map((row, index) => (
                <div key={row.root_letter} className="root-row">
                  <span>{index + 1}</span>
                  <strong>{row.root_letter.toUpperCase()}</strong>
                  <div className="bar-track">
                    <span style={{ width: `${Math.max(3, Number(row.win_rate) * 100)}%` }} />
                  </div>
                  <small>
                    {percent(Number(row.win_rate))} · {fixed(Number(row.average_mistakes))} mistakes
                  </small>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {activeTab === "about" && <About />}
    </main>
  );
}

function ChartPanel({
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
    <div className="panel chart-panel">
      <div className="section-heading">
        <BarChart3 size={20} />
        <h2>{title}</h2>
      </div>
      {rows.length ? (
        <ResponsiveContainer width="100%" height={260}>
          {chart === "line" ? (
            <LineChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263542" />
              <XAxis dataKey="label" stroke="#9fb0be" />
              <YAxis stroke="#9fb0be" unit={unit} />
              <Tooltip contentStyle={{ background: "#101821", border: "1px solid #263542" }} />
              <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          ) : (
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263542" />
              <XAxis dataKey="label" stroke="#9fb0be" />
              <YAxis stroke="#9fb0be" unit={unit} />
              <Tooltip contentStyle={{ background: "#101821", border: "1px solid #263542" }} />
              <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      ) : (
        <div className="empty-state">No rows available.</div>
      )}
    </div>
  );
}

export default App;

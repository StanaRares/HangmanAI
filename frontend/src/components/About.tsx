type ResearchStat = {
  length: number;
  win_rate: number;
  best_first_guess: string;
};

type Props = {
  vocabularySize?: number;
  trainedTrees?: number;
  hardestLength?: ResearchStat | null;
  easiestLength?: ResearchStat | null;
};

function integer(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "n/a";
}

function percent(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
}

export function About({ vocabularySize, trainedTrees, hardestLength, easiestLength }: Props) {
  return (
    <section className="research-page">
      <header className="research-hero">
        <p>The question</p>
        <h1>Is E really the best first guess?</h1>
      </header>

      <article className="research-block">
        <span>01</span>
        <div>
          <h2>The experiment</h2>
          <p>
            A separate decision tree is trained for every word length. The player is the tree
            itself: a node selects one letter, and the observed Hangman reveal pattern chooses the
            next branch.
          </p>
        </div>
      </article>

      <article className="research-block">
        <span>02</span>
        <div>
          <h2>The vocabulary</h2>
          <p>
            The English word forms come from wordfreq, cleaned into lowercase ASCII Hangman words
            and grouped by exact length. The current dataset contains {integer(vocabularySize)} word
            forms and {trainedTrees ?? 0} trained uniform trees.
          </p>
        </div>
      </article>

      <blockquote>
        Short words are dramatically harder because each guess reveals less structure while the
        mistake budget stays fixed.
      </blockquote>

      <article className="research-block">
        <span>03</span>
        <div>
          <h2>The result</h2>
          <p>
            In the current uniform run, the hardest length is{" "}
            {hardestLength ? `${hardestLength.length} letters at ${percent(hardestLength.win_rate)}` : "still loading"}.
            The easiest measured bucket is{" "}
            {easiestLength ? `${easiestLength.length} letters at ${percent(easiestLength.win_rate)}` : "still loading"}.
            First guesses shift across length: short buckets often open with A, mid-length buckets
            move to E, and longer buckets frequently start with I.
          </p>
        </div>
      </article>

      <article className="research-block">
        <span>04</span>
        <div>
          <h2>The limits</h2>
          <p>
            These are best trees found under recorded budgets and search profiles, not claims of
            mathematical global optimality. The UI reports real generated metrics and leaves missing
            root sweeps empty rather than filling in invented values.
          </p>
        </div>
      </article>
    </section>
  );
}

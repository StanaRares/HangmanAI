import { Brain, Database, GitBranch, Search, Split } from "lucide-react";

export function About() {
  return (
    <section className="about-grid">
      <article className="panel">
        <div className="section-heading">
          <Search size={20} />
          <h2>Research Question</h2>
        </div>
        <p>
          HangmanAI asks whether the best first guess and later decisions change when the hidden
          word length changes. The player is not an ensemble, neural model, or entropy agent: it is
          a stored Hangman decision tree selected by word length.
        </p>
      </article>
      <article className="panel">
        <div className="section-heading">
          <GitBranch size={20} />
          <h2>Multiway Tree</h2>
        </div>
        <p>
          Each internal node stores one letter guess. Branches are full Hangman reveal patterns, so
          a five-letter guess can split into outcomes such as 00000, 01000, or 10001 rather than a
          simple yes-or-no test.
        </p>
      </article>
      <article className="panel">
        <div className="section-heading">
          <Brain size={20} />
          <h2>Objective</h2>
        </div>
        <p>
          Tree construction optimizes actual Hangman performance: maximize win rate first, then
          prefer fewer wrong guesses, fewer total guesses, shallower traversal, and smaller trees.
          Greedy, bounded-lookahead, and near-exact profiles record their own limits.
        </p>
      </article>
      <article className="panel">
        <div className="section-heading">
          <Database size={20} />
          <h2>Vocabulary</h2>
        </div>
        <p>
          The research vocabulary is extracted from wordfreq using its English word-list iterator,
          cleaned to lowercase ASCII words, grouped by exact length, and stored with each word's
          Zipf frequency for later uniform versus frequency-weighted tree experiments.
        </p>
      </article>
      <article className="panel">
        <div className="section-heading">
          <Split size={20} />
          <h2>Root Letters</h2>
        </div>
        <p>
          The root analysis rebuilds a tree under each possible first letter and evaluates the
          resulting policy against every word of that length. That makes the traditional first-guess
          advice measurable instead of assumed.
        </p>
      </article>
    </section>
  );
}

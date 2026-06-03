// Порт ядра игры BADC (alpha_badc/game.py + config.py) на JS.
// Поддержан режим из GameConfig по умолчанию: 2 игрока, own_word="explode",
// opp_word="on", score_limit=4.0. Логика повторяет game.py один-в-один.

(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BADC = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const COLS = 9, ROWS = 5, WORD_LEN = 4;
  const LETTERS = ["a", "b", "c", "d"];
  const NUM_LETTERS = LETTERS.length;
  const EMPTY = -1, CORRUPT_ID = NUM_LETTERS; // 4 == "e"
  const DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];
  const WORDS_BY_ID = {
    1: [0, 1, 2, 3], 2: [1, 0, 3, 2], 3: [1, 2, 0, 3], 4: [3, 2, 1, 0],
  };

  const DEFAULT_CONFIG = {
    player_count: 2, own_word: "explode", opp_word: "on",
    score_limit: 4.0, max_turns: 600, cols: COLS, rows: ROWS,
  };

  function buildLines(cols, rows) {
    const lines = [];
    for (const [dx, dy] of DIRS)
      for (let row = 0; row < rows; row++)
        for (let col = 0; col < cols; col++) {
          const cells = []; let ok = true;
          for (let k = 0; k < WORD_LEN; k++) {
            const c = col + dx * k, r = row + dy * k;
            if (c < 0 || c >= cols || r < 0 || r >= rows) { ok = false; break; }
            cells.push(r * cols + c);
          }
          if (ok) lines.push(cells);
        }
    return lines;
  }

  function buildNeighbors(cols, rows) {
    const out = [];
    for (let i = 0; i < cols * rows; i++) {
      const c = i % cols, r = (i / cols) | 0, nb = [];
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) nb.push(nr * cols + nc);
        }
      out.push(nb);
    }
    return out;
  }

  class BADCGame {
    constructor(config) {
      this.config = Object.assign({}, DEFAULT_CONFIG, config || {});
      this.cols = this.config.cols;
      this.rows = this.config.rows;
      this.numCells = this.cols * this.rows;
      this.lines = buildLines(this.cols, this.rows);
      this.neighbors = buildNeighbors(this.cols, this.rows);
      this.words = [];
      for (let pid = 1; pid <= this.config.player_count; pid++)
        this.words.push(WORDS_BY_ID[pid].slice());
      this.reset();
    }

    reset() {
      this.board = new Int8Array(this.numCells).fill(EMPTY);
      this.scores = new Array(this.config.player_count).fill(0);
      this.turn = 0;
      this.winner = null;       // индекс игрока, -1 ничья, null = идёт
      this.winType = null;
      this.winningLine = null;
      this.lastMoveBy = new Array(this.config.player_count).fill(null);
      this.lastMoveTurn = new Array(this.config.player_count).fill(-1);
      return this;
    }

    get currentPlayer() { return this.turn % this.config.player_count; }
    get done() { return this.winner !== null; }

    emptyCells() {
      const out = [];
      for (let i = 0; i < this.numCells; i++) if (this.board[i] === EMPTY) out.push(i);
      return out;
    }

    legalMoves() {
      const moves = [];
      for (let i = 0; i < this.numCells; i++)
        if (this.board[i] === EMPTY)
          for (let l = 0; l < NUM_LETTERS; l++) moves.push([i, l]);
      return moves;
    }

    legalMask() {
      const mask = new Uint8Array(this.numCells * NUM_LETTERS);
      for (let i = 0; i < this.numCells; i++)
        if (this.board[i] === EMPTY)
          for (let l = 0; l < NUM_LETTERS; l++) mask[i * NUM_LETTERS + l] = 1;
      return mask;
    }

    static moveToAction(m) { return m[0] * NUM_LETTERS + m[1]; }
    static actionToMove(a) { return [(a / NUM_LETTERS) | 0, a % NUM_LETTERS]; }

    ruleFor(owner, mover) {
      return owner === mover ? this.config.own_word : this.config.opp_word;
    }

    _completions(board) {
      const out = [];
      for (const line of this.lines) {
        if (board[line[0]] === EMPTY) continue;
        for (let p = 0; p < this.config.player_count; p++) {
          const word = this.words[p]; let ok = true;
          for (let k = 0; k < WORD_LEN; k++)
            if (board[line[k]] !== word[k]) { ok = false; break; }
          if (ok) { out.push([p, line]); break; }
        }
      }
      return out;
    }

    hasLiveLine(board) {
      const b = board || this.board;
      for (const line of this.lines)
        for (let p = 0; p < this.config.player_count; p++) {
          const word = this.words[p]; let alive = true;
          for (let k = 0; k < WORD_LEN; k++) {
            const v = b[line[k]];
            if (v !== EMPTY && v !== word[k]) { alive = false; break; }
          }
          if (alive) return true;
        }
      return false;
    }

    apply(move) {
      if (this.done) throw new Error("game over");
      const [index, letter] = move;
      if (this.board[index] !== EMPTY) throw new Error("cell occupied");
      const mover = this.currentPlayer;

      this.board[index] = letter;
      this.lastMoveBy[mover] = index;
      this.lastMoveTurn[mover] = this.turn;
      const completions = this._completions(this.board);

      let instantWinner = null, instantLine = null;
      const clearSet = new Set(), corruptSet = new Set();

      for (const [player, line] of completions) {
        const rule = this.ruleFor(player, mover);
        if (rule === "off") {
          if (instantWinner === null || player === mover) {
            instantWinner = player; instantLine = line;
          }
        } else {
          if (player === mover) this.scores[mover] += 1.0;
          if (rule === "corrupt") {
            for (const idx of line) corruptSet.add(idx);
          } else {
            for (const idx of line) {
              clearSet.add(idx);
              if (rule === "explode") for (const nb of this.neighbors[idx]) clearSet.add(nb);
            }
          }
        }
      }

      if (instantWinner !== null) {
        this.winner = instantWinner; this.winningLine = instantLine; this.winType = "instant";
        return { type: "win", player: instantWinner, line: instantLine };
      }

      corruptSet.delete(index); clearSet.delete(index);
      for (const idx of corruptSet) this.board[idx] = CORRUPT_ID;
      for (const idx of clearSet) this.board[idx] = EMPTY; // стирание приоритетнее порчи

      if (this.config.score_limit !== null && this.config.score_limit !== undefined) {
        let best = null;
        for (let p = 0; p < this.config.player_count; p++)
          if (this.scores[p] >= this.config.score_limit &&
              (best === null || this.scores[p] > this.scores[best])) best = p;
        if (best !== null) {
          this.winner = best; this.winType = "limit"; this.winningLine = null;
          return { type: "win", player: best };
        }
      }

      this.turn += 1;
      if (this.emptyCells().length === 0 || this.turn >= this.config.max_turns ||
          !this.hasLiveLine()) this._finishByScore();
      return { type: "continue", cleared: [...clearSet], corrupted: [...corruptSet] };
    }

    _finishByScore() {
      let best = -1.0, winners = [];
      for (let p = 0; p < this.config.player_count; p++) {
        if (this.scores[p] > best) { best = this.scores[p]; winners = [p]; }
        else if (this.scores[p] === best) winners.push(p);
      }
      this.winner = winners.length === 1 ? winners[0] : -1;
      this.winType = "score"; this.winningLine = null;
    }

    resultFor(player) {
      if (!this.done) throw new Error("game not finished");
      if (this.winner === -1) return 0.0;
      return this.winner === player ? 1.0 : -1.0;
    }

    clone() {
      const g = Object.create(BADCGame.prototype);
      g.config = this.config; g.cols = this.cols; g.rows = this.rows;
      g.numCells = this.numCells; g.lines = this.lines; g.neighbors = this.neighbors;
      g.words = this.words;
      g.board = this.board.slice();
      g.scores = this.scores.slice();
      g.turn = this.turn; g.winner = this.winner; g.winType = this.winType;
      g.winningLine = this.winningLine;
      g.lastMoveBy = this.lastMoveBy.slice();
      g.lastMoveTurn = this.lastMoveTurn.slice();
      return g;
    }
  }

  return {
    BADCGame, COLS, ROWS, WORD_LEN, LETTERS, NUM_LETTERS,
    EMPTY, CORRUPT_ID, DIRS, WORDS_BY_ID, DEFAULT_CONFIG,
  };
});

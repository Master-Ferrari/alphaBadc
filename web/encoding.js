// Порт alpha_badc/encoding.py: кодирование позиции в тензор (rows, cols, C),
// разложенный в Float32Array в порядке channels_last: idx = (r*cols + c)*C + ch.
// Порядок каналов точно повторяет encode_state (проверено parity_test.js).

(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(require("./game.js"));
  else root.AZEncode = factory(root.BADC);
})(typeof self !== "undefined" ? self : this, function (BADC) {
  "use strict";
  const { WORD_LEN, NUM_LETTERS, EMPTY, CORRUPT_ID } = BADC;

  function numChannels() {
    return NUM_LETTERS + 1 + 1 + 2 * (WORD_LEN * NUM_LETTERS) + 2 + 1 + 2; // = 43
  }

  // Плоскости одного слова: WORD_LEN*NUM_LETTERS broadcast-карт.
  // planes[pos*NUM_LETTERS + letter] == 1 на всём поле.
  function wordPlaneFlags(word) {
    const flags = new Uint8Array(WORD_LEN * NUM_LETTERS);
    for (let pos = 0; pos < WORD_LEN; pos++) flags[pos * NUM_LETTERS + word[pos]] = 1;
    return flags;
  }

  // game: экземпляр BADCGame. Возвращает Float32Array(rows*cols*C).
  function encodeState(game) {
    const rows = game.rows, cols = game.cols, board = game.board;
    const mover = game.currentPlayer;
    const C = numChannels();
    const out = new Float32Array(rows * cols * C);

    // Каналовые константы.
    const moverFlags = wordPlaneFlags(game.words[mover]);
    const oppFlags = new Uint8Array(WORD_LEN * NUM_LETTERS);
    for (let p = 0; p < game.config.player_count; p++) {
      if (p === mover) continue;
      const f = wordPlaneFlags(game.words[p]);
      for (let i = 0; i < f.length; i++) if (f[i]) oppFlags[i] = 1;
    }

    const denom = game.config.score_limit ? game.config.score_limit : 8.0;
    const myScore = game.scores[mover] / denom;
    let oppScore = 0;
    for (let p = 0; p < game.config.player_count; p++)
      if (p !== mover) oppScore = Math.max(oppScore, game.scores[p]);
    oppScore /= denom;
    const progress = game.turn / Math.max(1, game.config.max_turns);

    const myLast = game.lastMoveBy[mover];
    let oppLast = null, bestTurn = -1;
    for (let p = 0; p < game.config.player_count; p++) {
      if (p === mover) continue;
      if (game.lastMoveTurn[p] > bestTurn) { bestTurn = game.lastMoveTurn[p]; oppLast = game.lastMoveBy[p]; }
    }

    for (let cell = 0; cell < rows * cols; cell++) {
      const v = board[cell];
      const base = cell * C;
      let ch = 0;
      for (let l = 0; l < NUM_LETTERS; l++) out[base + ch++] = v === l ? 1 : 0;
      out[base + ch++] = v === CORRUPT_ID ? 1 : 0;
      out[base + ch++] = v === EMPTY ? 1 : 0;
      for (let i = 0; i < moverFlags.length; i++) out[base + ch++] = moverFlags[i];
      for (let i = 0; i < oppFlags.length; i++) out[base + ch++] = oppFlags[i];
      out[base + ch++] = myScore;
      out[base + ch++] = oppScore;
      out[base + ch++] = progress;
      out[base + ch++] = cell === myLast ? 1 : 0;
      out[base + ch++] = cell === oppLast ? 1 : 0;
    }
    return out;
  }

  return { encodeState, numChannels };
});

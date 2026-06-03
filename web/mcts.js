// Порт PUCT-MCTS (alpha_badc/mcts.py). Для веб-оппонента используется как в
// arena.NetAgent: без dirichlet-шума и с argmax по визитам (temperature=0).

(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(require("./game.js"), require("./encoding.js"));
  else root.AZMcts = factory(root.BADC, root.AZEncode);
})(typeof self !== "undefined" ? self : this, function (BADC, AZEncode) {
  "use strict";
  const { BADCGame } = BADC;
  const { encodeState } = AZEncode;

  class Node {
    constructor(game) {
      this.game = game;
      this.player = game.currentPlayer;
      this.isExpanded = false;
      this.legal = [];
      this.P = new Map();
      this.N = new Map();
      this.W = new Map();
      this.children = new Map();
    }
    q(a) { const n = this.N.get(a) || 0; return n === 0 ? 0 : this.W.get(a) / n; }
    visitSum() { let s = 0; for (const v of this.N.values()) s += v; return s; }
  }

  class MCTS {
    constructor(net, cfg) {
      this.net = net;
      this.cfg = Object.assign({ num_simulations: 80, c_puct: 1.5 }, cfg || {});
      this.actionSize = net.actionSize;
    }

    // Вернуть Int32Array визитов по действиям.
    run(rootGame) {
      const root = new Node(rootGame.clone());
      this._expand(root);
      for (let i = 0; i < this.cfg.num_simulations; i++) this._simulate(root);
      const visits = new Float64Array(this.actionSize);
      for (const [a, n] of root.N) visits[a] = n;
      return visits;
    }

    _simulate(root) {
      const path = [];
      let node = root;
      while (node.isExpanded && !node.game.done) {
        const action = this._select(node);
        path.push([node, action]);
        let child = node.children.get(action);
        if (child === undefined) {
          const cg = node.game.clone();
          cg.apply(BADCGame.actionToMove(action));
          child = new Node(cg);
          node.children.set(action, child);
        }
        node = child;
      }

      let leafValue, leafPlayer;
      if (node.game.done) {
        leafValue = node.game.winner !== -1 ? node.game.resultFor(node.player) : 0.0;
        leafPlayer = node.player;
      } else {
        leafValue = this._expand(node);
        leafPlayer = node.player;
      }

      for (const [n, a] of path) {
        const signed = n.player === leafPlayer ? leafValue : -leafValue;
        n.N.set(a, (n.N.get(a) || 0) + 1);
        n.W.set(a, (n.W.get(a) || 0) + signed);
      }
    }

    _select(node) {
      const total = node.visitSum();
      const sqrtTotal = Math.sqrt(total + 1e-8);
      let bestA = -1, bestU = -Infinity;
      for (const a of node.legal) {
        const p = node.P.get(a) || 0;
        const u = node.q(a) + this.cfg.c_puct * p * sqrtTotal / (1 + (node.N.get(a) || 0));
        if (u > bestU) { bestU = u; bestA = a; }
      }
      return bestA;
    }

    _expand(node) {
      const { probs, value } = this.net.predict(encodeState(node.game));
      const mask = node.game.legalMask();
      node.legal = [];
      let s = 0;
      for (let a = 0; a < mask.length; a++) if (mask[a]) { node.legal.push(a); s += probs[a]; }
      for (const a of node.legal) {
        const p = s > 1e-8 ? probs[a] / s : 1 / node.legal.length;
        node.P.set(a, p);
        node.N.set(a, 0);
        node.W.set(a, 0);
      }
      node.isExpanded = true;
      return value;
    }
  }

  function selectAction(visits, temperature) {
    if (temperature <= 1e-3) {
      let best = 0, bi = 0;
      for (let i = 0; i < visits.length; i++) if (visits[i] > best) { best = visits[i]; bi = i; }
      return bi;
    }
    const logits = new Float64Array(visits.length);
    let total = 0;
    for (let i = 0; i < visits.length; i++) { logits[i] = Math.pow(visits[i], 1 / temperature); total += logits[i]; }
    if (total <= 0) { // argmax
      let best = 0, bi = 0;
      for (let i = 0; i < visits.length; i++) if (visits[i] > best) { best = visits[i]; bi = i; }
      return bi;
    }
    let r = Math.random() * total;
    for (let i = 0; i < logits.length; i++) { r -= logits[i]; if (r <= 0) return i; }
    return logits.length - 1;
  }

  return { MCTS, selectAction };
});

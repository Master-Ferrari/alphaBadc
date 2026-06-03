// Склейка UI: партия BADC + сеть (MCTS) целиком в браузере, без сервера.
// Управление: сперва выбираешь клетку, затем букву — клик по букве делает ход.
(function () {
  "use strict";
  const { BADCGame, LETTERS } = BADC;
  const { MCTS, selectAction } = AZMcts;
  const CHARS = LETTERS.concat(["e"]);

  let net = null, mcts = null, game = null;
  let humanSeat = 0, selectedCell = null, thinking = false;

  function yourTurn() { return !game.done && game.currentPlayer === humanSeat; }

  function renderWords() {
    const tiles = (word) => word.map((l) => `<span class="tile ${LETTERS[l]}">${LETTERS[l]}</span>`).join("");
    document.getElementById("myword").innerHTML = tiles(game.words[humanSeat]);
    document.getElementById("oppword").innerHTML = tiles(game.words[1 - humanSeat]);
  }

  function netMove() {
    while (!game.done && game.currentPlayer !== humanSeat) {
      const visits = mcts.run(game);
      game.apply(BADCGame.actionToMove(selectAction(visits, 0.0)));
    }
  }

  function scheduleNet() {
    thinking = true; selectedCell = null; render();
    setTimeout(() => { netMove(); thinking = false; render(); }, 20);
  }

  function selectCell(cell) {
    if (!game || !yourTurn() || thinking) return;
    if (game.board[cell] !== BADC.EMPTY) return;
    selectedCell = (selectedCell === cell) ? null : cell;  // повторный клик снимает выбор
    render();
  }

  function commitLetter(letter) {
    if (!game || !yourTurn() || thinking || selectedCell === null) return;
    game.apply([selectedCell, letter]);
    selectedCell = null;
    render();
    if (!game.done && !yourTurn()) scheduleNet();
  }

  window.showSetup = function () {
    document.getElementById("game").hidden = true;
    document.getElementById("setup").hidden = false;
  };

  window.startGame = function () {
    if (!net) return;
    humanSeat = +document.querySelector("input[name=seat]:checked").value;
    mcts = new MCTS(net, { num_simulations: +document.getElementById("sims").value });
    game = new BADCGame();
    selectedCell = null;
    document.getElementById("setup").hidden = true;
    document.getElementById("game").hidden = false;
    // ширина арены = ширина поля: cols клеток по 54px + зазоры 4px
    document.getElementById("arena").style.width = (game.cols * 54 + (game.cols - 1) * 4) + "px";
    renderWords(); render();
    if (!yourTurn()) scheduleNet();
  };

  function render() {
    if (!game) return;

    // поле
    const g = document.getElementById("grid");
    g.style.gridTemplateColumns = `repeat(${game.cols},1fr)`;
    g.innerHTML = "";
    const winLine = new Set(game.winningLine || []);
    for (let i = 0; i < game.board.length; i++) {
      const v = game.board[i];
      const b = document.createElement("button");
      b.className = "cell" + (winLine.has(i) ? " win" : "") + (selectedCell === i ? " sel" : "");
      if (v >= 0) { const ch = CHARS[v]; b.textContent = ch; b.classList.add(ch); }
      b.disabled = thinking || !yourTurn() || v >= 0;
      b.onclick = () => selectCell(i);
      g.appendChild(b);
    }

    // выбор буквы
    const picker = document.getElementById("picker");
    picker.innerHTML = "";
    const canPick = yourTurn() && !thinking && selectedCell !== null;
    LETTERS.forEach((ch, i) => {
      const b = document.createElement("button");
      b.className = "pick " + ch;
      b.textContent = ch;
      b.disabled = !canPick;
      b.onclick = () => commitLetter(i);
      picker.appendChild(b);
    });

    // счёт и подсветка активной стороны (объединено с карточками игроков)
    const fmt = (x) => (Math.round(x * 10) / 10).toString();
    const myOn = yourTurn(), oppOn = !game.done && !myOn;
    document.getElementById("mypts").textContent = fmt(game.scores[humanSeat]);
    document.getElementById("opppts").textContent = fmt(game.scores[1 - humanSeat]);
    document.getElementById("pcard-me").classList.toggle("on", myOn);
    document.getElementById("pcard-opp").classList.toggle("on", oppOn);

    // единый текстовый блок снизу: подсказка/исход + номер хода и лимит
    const limit = game.config.score_limit;
    const meta = `ход ${game.turn}` + (limit ? ` · до ${fmt(limit)} очков` : "");
    let prompt;
    if (game.done) {
      if (game.winner === -1) prompt = "Ничья.";
      else if (game.winner === humanSeat) prompt = `Вы выиграли (${game.winType})`;
      else prompt = `Сеть выиграла (${game.winType})`;
    } else if (thinking) {
      prompt = "Ход сети…";
    } else if (!yourTurn()) {
      prompt = "";
    } else {
      prompt = selectedCell === null ? "Выберите клетку" : "Выберите букву";
    }
    document.getElementById("status").textContent = prompt ? `${prompt} · ${meta}` : meta;
  }

  AZNet.load("").then((n) => {
    net = n;
    document.getElementById("boot").textContent = "";
    document.getElementById("playbtn").disabled = false;  // экран настроек показан со старта
  }).catch((e) => {
    document.getElementById("boot").textContent = "Ошибка загрузки модели: " + e;
  });
})();

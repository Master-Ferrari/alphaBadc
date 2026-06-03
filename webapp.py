"""Минимальное веб-приложение: игра человека против обученного чекпоинта.

Запуск:
    python webapp.py --checkpoint checkpoints/az_badc_v3_latest.weights.h5

Затем открыть http://localhost:8000 в браузере. Зависимостей сверх проекта нет —
используется встроенный http.server. Человек ходит за одного игрока, чекпоинт
(сеть + MCTS) — за другого.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from alpha_badc.arena import NetAgent
from alpha_badc.config import AZConfig, GameConfig, LETTERS
from alpha_badc.encoding import action_size, num_channels
from alpha_badc.game import BADCGame, CORRUPT_ID, EMPTY
from alpha_badc.network import AZNetwork

CHARS = LETTERS + ["e"]

# Заполняется в main().
NET_AGENT: NetAgent | None = None
GAME_CFG: GameConfig | None = None
LOCK = threading.Lock()

# Состояние одной партии (одна на сервер — это «минимально»).
STATE: dict = {"game": None, "human_seat": 0}


def new_game(human_seat: int) -> None:
    game = BADCGame(GAME_CFG)
    STATE["game"] = game
    STATE["human_seat"] = human_seat
    _maybe_net_move(game)


def _maybe_net_move(game: BADCGame) -> None:
    """Пока ходить должна сеть и партия не окончена — она ходит."""
    while not game.done and game.current_player != STATE["human_seat"]:
        action = NET_AGENT(game)
        game.apply(BADCGame.action_to_move(action))


def game_state() -> dict:
    game: BADCGame = STATE["game"]
    if game is None:
        return {"started": False}
    return {
        "started": True,
        "rows": game.rows,
        "cols": game.cols,
        "board": [int(v) for v in game.board],
        "chars": CHARS,
        "letters": LETTERS,
        "scores": [round(s, 1) for s in game.scores],
        "turn": game.turn,
        "current_player": game.current_player,
        "human_seat": STATE["human_seat"],
        "your_turn": (not game.done) and game.current_player == STATE["human_seat"],
        "done": bool(game.done),
        "winner": game.winner,
        "win_type": game.win_type,
        "winning_line": game.winning_line,
    }


def human_move(cell: int, letter: int) -> dict:
    game: BADCGame = STATE["game"]
    if game is None:
        return {"error": "no game"}
    if game.done:
        return {"error": "game over"}
    if game.current_player != STATE["human_seat"]:
        return {"error": "not your turn"}
    if cell < 0 or cell >= game.num_cells or game.board[cell] != EMPTY:
        return {"error": "illegal cell"}
    if letter < 0 or letter >= len(LETTERS):
        return {"error": "illegal letter"}
    game.apply((cell, letter))
    _maybe_net_move(game)
    return game_state()


INDEX_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>AlphaBADC — играй против чекпоинта</title>
<style>
  body{font-family:system-ui,Segoe UI,sans-serif;background:#15151c;color:#e6e6ee;margin:0;padding:24px;text-align:center}
  h1{font-size:20px;font-weight:600}
  #grid{display:inline-grid;gap:4px;margin:16px auto}
  .cell{width:56px;height:56px;border:1px solid #3a3a48;background:#23232e;border-radius:8px;
        font-size:26px;font-weight:700;cursor:pointer;color:#e6e6ee;transition:.1s}
  .cell:hover:enabled{background:#2f2f3e}
  .cell:disabled{cursor:default}
  .win{outline:3px solid #d8c84a}
  .a{color:#00c8f0}.b{color:#f0dc00}.c{color:#50a0ff}.d{color:#f09600}.e{color:#888}
  .bar{margin:8px}
  label{margin:0 8px;cursor:pointer}
  button.act{background:#3a5bd8;color:#fff;border:0;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:14px}
  #info{margin:12px;font-size:15px;min-height:22px}
  #status{font-weight:600}
</style></head>
<body>
<h1>AlphaBADC — человек против чекпоинта</h1>
<div class="bar">
  Буква:
  <span id="letters"></span>
</div>
<div class="bar">
  Место человека:
  <label><input type="radio" name="seat" value="0" checked> ходит первым</label>
  <label><input type="radio" name="seat" value="1"> ходит вторым</label>
  <button class="act" onclick="newGame()">Новая игра</button>
</div>
<div id="info"></div>
<div><div id="grid"></div></div>
<div id="status"></div>

<script>
let state=null, letter=0;
const LCHARS=["a","b","c","d"];

function renderLetters(){
  const box=document.getElementById('letters'); box.innerHTML='';
  LCHARS.forEach((ch,i)=>{
    const id='l'+i;
    box.insertAdjacentHTML('beforeend',
      `<label><input type="radio" name="letter" value="${i}" ${i===letter?'checked':''}> ${ch}</label>`);
  });
  box.querySelectorAll('input').forEach(inp=>inp.onchange=()=>letter=+inp.value);
}

async function api(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})});
  return r.json();
}

async function newGame(){
  const seat=+document.querySelector('input[name=seat]:checked').value;
  state=await api('/new',{human_seat:seat});
  render();
}

async function play(cell){
  if(!state||!state.your_turn) return;
  const res=await api('/move',{cell:cell,letter:letter});
  if(res.error){document.getElementById('status').textContent='⚠ '+res.error;return;}
  state=res; render();
}

function render(){
  if(!state||!state.started) return;
  const g=document.getElementById('grid');
  g.style.gridTemplateColumns=`repeat(${state.cols},56px)`;
  g.innerHTML='';
  const winLine=new Set(state.winning_line||[]);
  for(let i=0;i<state.board.length;i++){
    const v=state.board[i];
    const b=document.createElement('button');
    b.className='cell'+(winLine.has(i)?' win':'');
    if(v>=0){const ch=state.chars[v]; b.textContent=ch; b.classList.add(ch);}
    b.disabled=!state.your_turn||v>=0;
    b.onclick=()=>play(i);
    g.appendChild(b);
  }
  const seatTxt='Вы — игрок '+(state.human_seat+1);
  document.getElementById('info').textContent=
    `${seatTxt} · ход ${state.turn} · счёт [${state.scores.join(', ')}]`;
  const s=document.getElementById('status');
  if(state.done){
    if(state.winner===-1) s.textContent='Ничья.';
    else if(state.winner===state.human_seat) s.textContent='🎉 Вы выиграли! ('+state.win_type+')';
    else s.textContent='Чекпоинт выиграл. ('+state.win_type+')';
  } else {
    s.textContent=state.your_turn?'Ваш ход: выберите букву и кликните пустую клетку.'
                                 :'Ход чекпоинта…';
  }
}

renderLetters();
newGame();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # тише в консоли
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            with LOCK:
                self._json(game_state())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        with LOCK:
            if self.path == "/new":
                new_game(int(payload.get("human_seat", 0)))
                self._json(game_state())
            elif self.path == "/move":
                self._json(human_move(int(payload["cell"]), int(payload["letter"])))
            else:
                self._json({"error": "not found"}, 404)


def main() -> None:
    global NET_AGENT, GAME_CFG
    parser = argparse.ArgumentParser(description="Web UI: человек против чекпоинта AlphaBADC.")
    parser.add_argument("--checkpoint", default="checkpoints/az_badc_v3_latest.weights.h5")
    parser.add_argument("--sims", type=int, default=80, help="число симуляций MCTS на ход")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--head-kernel", type=int, default=3,
                        help="размер ядра в головах (должен совпадать с чекпоинтом; претрейн — 4)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    GAME_CFG = GameConfig()
    az_cfg = AZConfig(channels=args.channels, blocks=args.blocks, head_kernel=args.head_kernel)
    net = AZNetwork((GAME_CFG.rows, GAME_CFG.cols, num_channels()), action_size(GAME_CFG), az_cfg)
    # Чекпоинты обучены с меньшим числом входных каналов (без плоскостей
    # последнего хода) — переносим совпадающие, новые обнуляем.
    net.load_partial(args.checkpoint, zero_new_channels=True)
    NET_AGENT = NetAgent(net, az_cfg, simulations=args.sims)
    print(f"[web] loaded {args.checkpoint} (sims={args.sims})")

    new_game(0)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web] открой http://{args.host}:{args.port}  (Ctrl+C для остановки)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] остановлено")


if __name__ == "__main__":
    main()

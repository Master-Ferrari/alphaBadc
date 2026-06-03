// Проверка паритета JS-инференса (net.js) с эталоном из Python (parity.json).
// Запуск:  node web/parity_test.js
const fs = require("fs");
const path = require("path");
const AZNet = require("./net.js");

const dir = __dirname;
const meta = JSON.parse(fs.readFileSync(path.join(dir, "model.json")));
const bin = fs.readFileSync(path.join(dir, "model.bin"));
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const net = new AZNet(meta, buf);

const { BADCGame } = require("./game.js");
const { encodeState } = require("./encoding.js");

const parity = JSON.parse(fs.readFileSync(path.join(dir, "parity.json")));
let maxProb = 0, maxVal = 0, maxEnc = 0;
parity.cases.forEach((c, i) => {
  // 1) кодировка: воспроизводим ходы в JS и сравниваем тензор с эталоном Python
  const g = new BADCGame();
  for (const a of c.actions) g.apply(BADCGame.actionToMove(a));
  const st = encodeState(g);
  let de = 0;
  for (let j = 0; j < st.length; j++) de = Math.max(de, Math.abs(st[j] - c.state[j]));
  maxEnc = Math.max(maxEnc, de);

  // 2) инференс на JS-кодировке (а не на эталонном state) — сквозная проверка
  const { probs, value } = net.predict(st);
  let dp = 0;
  for (let j = 0; j < probs.length; j++) dp = Math.max(dp, Math.abs(probs[j] - c.probs[j]));
  const dv = Math.abs(value - c.value);
  maxProb = Math.max(maxProb, dp);
  maxVal = Math.max(maxVal, dv);
  console.log(`case ${i}: max|Δenc|=${de.toExponential(2)}  max|Δprob|=${dp.toExponential(2)}  |Δvalue|=${dv.toExponential(2)}`);
});
console.log(`\nИТОГ: max|Δenc|=${maxEnc.toExponential(2)}  max|Δprob|=${maxProb.toExponential(2)}  max|Δvalue|=${maxVal.toExponential(2)}`);
const ok = maxEnc < 1e-4 && maxProb < 1e-4 && maxVal < 1e-4;
console.log(ok ? "PARITY OK ✓" : "PARITY FAIL ✗");
process.exit(ok ? 0 : 1);

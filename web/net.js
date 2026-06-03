// Ручной forward сети AlphaBADC на чистом JS (без ML-рантайма).
// Архитектура и порядок весов — см. scripts/export_web.py.
//
// Тензоры карт признаков хранятся как Float32Array длиной H*W*C в порядке
// channels_last (C-order): idx = (h*W + w)*C + c — так же, как Keras Flatten.
// Веса conv — в порядке Keras (kh, kw, Cin, Cout).

(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AZNet = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function relu(a) {
    for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
    return a;
  }

  // Conv2D с паддингом "same" (stride 1), как в TensorFlow.
  // in: (H,W,Cin) Float32Array; w: (kh,kw,Cin,Cout); b: (Cout)
  function conv2dSame(input, H, W, Cin, w, b, kh, kw, Cout) {
    const out = new Float32Array(H * W * Cout);
    const padTop = (kh - 1) >> 1;   // floor((kh-1)/2) — правило TF SAME
    const padLeft = (kw - 1) >> 1;
    for (let h = 0; h < H; h++) {
      for (let x = 0; x < W; x++) {
        const obase = (h * W + x) * Cout;
        for (let co = 0; co < Cout; co++) out[obase + co] = b[co];
        for (let kr = 0; kr < kh; kr++) {
          const ih = h + kr - padTop;
          if (ih < 0 || ih >= H) continue;
          for (let kc = 0; kc < kw; kc++) {
            const iw = x + kc - padLeft;
            if (iw < 0 || iw >= W) continue;
            const ibase = (ih * W + iw) * Cin;
            const wbase = ((kr * kw + kc) * Cin) * Cout;
            for (let ci = 0; ci < Cin; ci++) {
              const v = input[ibase + ci];
              if (v === 0) continue;
              const wb = wbase + ci * Cout;
              for (let co = 0; co < Cout; co++) out[obase + co] += v * w[wb + co];
            }
          }
        }
      }
    }
    return out;
  }

  // Dense: in (Nin), w (Nin, Nout) [Keras], b (Nout) -> (Nout)
  function dense(input, w, b, Nin, Nout) {
    const out = new Float32Array(Nout);
    for (let o = 0; o < Nout; o++) out[o] = b[o];
    for (let i = 0; i < Nin; i++) {
      const v = input[i];
      if (v === 0) continue;
      const wb = i * Nout;
      for (let o = 0; o < Nout; o++) out[o] += v * w[wb + o];
    }
    return out;
  }

  function softmax(logits) {
    let m = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
    let s = 0;
    const out = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - m); s += out[i]; }
    for (let i = 0; i < logits.length; i++) out[i] /= s;
    return out;
  }

  function AZNet(meta, buffer) {
    this.meta = meta;
    const f32 = new Float32Array(buffer);
    const W = {};
    for (const [name, info] of Object.entries(meta.weights)) {
      W[name] = f32.subarray(info.offset, info.offset + info.size);
    }
    this.W = W;
    this.H = meta.input_shape[0];
    this.Wd = meta.input_shape[1];
    this.Cin = meta.input_shape[2];
    this.C = meta.channels;
    this.blocks = meta.blocks;
    this.k = meta.kernel;
    this.hk = meta.head_kernel;
    this.actionSize = meta.action_size;
  }

  // state: Float32Array (H*W*Cin) в порядке channels_last.
  // Возвращает { probs: Float32Array(actionSize), value: number }.
  AZNet.prototype.predict = function (state) {
    const { W, H, Wd, Cin, C, k, hk, blocks } = this;
    let x = relu(conv2dSame(state, H, Wd, Cin, W["stem.w"], W["stem.b"], k, k, C));
    for (let i = 0; i < blocks; i++) {
      const a = relu(conv2dSame(x, H, Wd, C, W[`res${i}.a.w`], W[`res${i}.a.b`], k, k, C));
      const bm = conv2dSame(a, H, Wd, C, W[`res${i}.b.w`], W[`res${i}.b.b`], k, k, C);
      for (let j = 0; j < bm.length; j++) { bm[j] += x[j]; if (bm[j] < 0) bm[j] = 0; }
      x = bm;
    }
    // policy head
    const p = relu(conv2dSame(x, H, Wd, C, W["policy.conv.w"], W["policy.conv.b"], hk, hk, 32));
    const logits = dense(p, W["policy.fc.w"], W["policy.fc.b"], H * Wd * 32, this.actionSize);
    // value head
    const v = relu(conv2dSame(x, H, Wd, C, W["value.conv.w"], W["value.conv.b"], hk, hk, 32));
    const h1 = relu(dense(v, W["value.fc1.w"], W["value.fc1.b"], H * Wd * 32, 128));
    const vo = dense(h1, W["value.fc2.w"], W["value.fc2.b"], 128, 1);
    return { probs: softmax(logits), value: Math.tanh(vo[0]) };
  };

  // Браузер: загрузить из baseUrl (model.json + model.bin).
  AZNet.load = async function (baseUrl) {
    baseUrl = baseUrl || "";
    const meta = await (await fetch(baseUrl + "model.json")).json();
    const buf = await (await fetch(baseUrl + "model.bin")).arrayBuffer();
    return new AZNet(meta, buf);
  };

  AZNet.softmax = softmax;
  return AZNet;
});

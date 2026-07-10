// Differentiable ops (DESIGN §7): forward via Backend kernels, backward as a
// closure capturing only what it needs. All ops here are exercised by the
// finite-difference grad-checks in test/gradcheck.test.ts.
import { B } from "../backend/index.js";
import { Tensor, opOutput, addInto } from "./tensor.js";
const prod = (s) => s.reduce((a, b) => a * b, 1);
/** a + b, with b's shape a suffix of a's shape (bias / row broadcast). */
export function add(a, b) {
    const nd = B().add(a.nd, b.nd);
    return opOutput(nd, [a, b], (out) => () => {
        const g = out.grad;
        if (a.requiresGrad)
            addInto(a.ensureGrad(), g);
        if (b.requiresGrad) {
            if (b.size === a.size)
                addInto(b.ensureGrad(), g);
            // Broadcasting rule: in the forward pass b (a bias / row vector) was COPIED
            // across the broadcast axis. The adjoint of a copy is a SUM — every position
            // b was copied to contributes to b's gradient — so we sum g back down to b's
            // shape ("sumTo"). Getting this right is the discipline that prevents shape bugs.
            else
                addInto(b.ensureGrad(), B().sumEvery(g, b.size));
        }
    });
}
/** s · a for scalar constant s. */
export function scale(a, s) {
    const nd = B().scale(a.nd, s);
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().scale({ data: out.grad, shape: out.shape }, s).data);
    });
}
/** a @ b. b may be 2-D while a is batched (weight broadcast).
 *
 *  Backward rule for C = A @ B — the workhorse gradient behind every Linear layer
 *  and all of attention. Given the upstream grad g = dL/dC, matrix calculus gives:
 *      dL/dA = g @ Bᵀ        dL/dB = Aᵀ @ g
 *  Intuition: each is "the OTHER input, transposed, times the incoming gradient" —
 *  the only arrangement that makes the shapes match (dL/dA must match A). We pass
 *  transposeA/transposeB flags so the backend never allocates a transposed copy. */
export function matmul(a, b) {
    const nd = B().matmul(a.nd, b.nd);
    return opOutput(nd, [a, b], (out) => () => {
        const g = { data: out.grad, shape: out.shape };
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().matmul(g, b.nd, { transposeB: true }).data);
        if (b.requiresGrad) {
            let gb = B().matmul(a.nd, g, { transposeA: true }); // [...batch, K, N]
            if (b.shape.length === 2 && a.shape.length > 2)
                gb = B().sumBatch(gb, b.shape);
            addInto(b.ensureGrad(), gb.data);
        }
    });
}
/** a @ bᵀ: a [.., M, K], b [.., N, K] (or [N, K] broadcast) → [.., M, N]. */
export function matmulT(a, b) {
    const nd = B().matmul(a.nd, b.nd, { transposeB: true });
    return opOutput(nd, [a, b], (out) => () => {
        const g = { data: out.grad, shape: out.shape };
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().matmul(g, b.nd).data);
        if (b.requiresGrad) {
            let gb = B().matmul(g, a.nd, { transposeA: true }); // gᵀ @ a → [.., N, K]
            if (b.shape.length === 2 && a.shape.length > 2)
                gb = B().sumBatch(gb, b.shape);
            addInto(b.ensureGrad(), gb.data);
        }
    });
}
export function gelu(a) {
    const nd = B().gelu(a.nd);
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().geluBackward(a.nd, { data: out.grad, shape: out.shape }).data);
    });
}
export function softmaxLastDim(a) {
    const nd = B().softmaxLastDim(a.nd);
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().softmaxBackward(out.nd, { data: out.grad, shape: out.shape }).data);
    });
}
export function layerNorm(x, w, b, eps = 1e-5) {
    const { y, mean, rstd } = B().layerNorm(x.nd, w.nd, b.nd, eps);
    return opOutput(y, [x, w, b], (out) => () => {
        const g = { data: out.grad, shape: out.shape };
        const { dx, dw, db } = B().layerNormBackward(x.nd, w.nd, mean, rstd, g);
        if (x.requiresGrad)
            addInto(x.ensureGrad(), dx.data);
        if (w.requiresGrad)
            addInto(w.ensureGrad(), dw);
        if (b.requiresGrad)
            addInto(b.ensureGrad(), db);
    });
}
/** Row-gather from an embedding table; out is reshaped to outShape. */
export function embedding(ids, table, outShape) {
    const g = B().gatherRows(table.nd, ids);
    if (g.data.length !== prod(outShape))
        throw new Error(`embedding: bad outShape [${outShape}]`);
    return opOutput({ data: g.data, shape: [...outShape] }, [table], (out) => () => {
        if (table.requiresGrad)
            B().scatterAddRows(table.ensureGrad(), table.shape[1], ids, out.grad);
    });
}
/** View with a new shape (data buffer is shared; tensors are always contiguous). */
export function reshape(a, shape) {
    if (prod(shape) !== a.size)
        throw new Error(`reshape: [${a.shape}] → [${shape}]`);
    return opOutput({ data: a.data, shape: [...shape] }, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), out.grad);
    });
}
/** Swap dims 1 and 2 of a 4-D tensor (attention head split/merge). */
export function transpose12(a) {
    const nd = B().transpose12(a.nd);
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().transpose12({ data: out.grad, shape: out.shape }).data);
    });
}
/** Causal mask on [.., T, T] attention scores: j > i → -1e30. */
export function causalMask(a, T) {
    const nd = B().causalMask(a.nd, T);
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad) {
            // .slice() COPIES the upstream grad before we zero the masked positions:
            // out.grad is shared with other consumers, so mutating it in place would
            // corrupt their gradient. Masked entries got -1e30 in forward → 0 grad here.
            const gm = { data: out.grad.slice(), shape: out.shape };
            B().causalMaskZeroGrad(gm, T);
            addInto(a.ensureGrad(), gm.data);
        }
    });
}
/** Mean cross-entropy over [N, V] logits vs integer targets. Fused & stable.
 *
 *  Forward: loss = mean( logsumexp(z) − z[target] ) — the negative log-probability
 *  the model assigns to the correct token, computed via logsumexp so it never forms
 *  a softmax that could overflow. Backward has an especially clean closed form:
 *      dL/dz = (softmax(z) − onehot(target)) / N
 *  i.e. "push the right token's probability up, all others down." Fusing forward and
 *  backward lets ceBackward reuse the logsumexp (`lse`) from the forward pass. */
export function crossEntropyLogits(logits, targets) {
    if (logits.shape.length !== 2)
        throw new Error("crossEntropyLogits: expected [N, V]");
    const { loss, lse } = B().ceForward(logits.nd, targets);
    const nd = B().from([loss], [1]);
    return opOutput(nd, [logits], (out) => () => {
        if (logits.requiresGrad)
            addInto(logits.ensureGrad(), B().ceBackward(logits.nd, lse, targets, out.grad[0]).data);
    });
}
/** Inverted dropout: scales kept activations by 1/(1-p). Call only when p > 0. */
export function dropout(a, p, rng) {
    const keep = 1 / (1 - p);
    const mask = new a.data.constructor(a.size);
    for (let i = 0; i < mask.length; i++)
        mask[i] = rng.random() < p ? 0 : keep;
    const nd = B().mul(a.nd, { data: mask, shape: [...a.shape] });
    return opOutput(nd, [a], (out) => () => {
        if (a.requiresGrad)
            addInto(a.ensureGrad(), B().mul({ data: out.grad, shape: out.shape }, { data: mask, shape: out.shape }).data);
    });
}

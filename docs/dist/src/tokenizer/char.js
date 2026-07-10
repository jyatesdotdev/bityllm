// Char-level tokenizer (DESIGN §11): vocab = sorted unique chars of the corpus.
// The vocab travels inside checkpoints so inference reconstructs it exactly.
export class CharTokenizer {
    vocab;
    stoi;
    constructor(vocab) {
        this.vocab = vocab;
        this.stoi = new Map(vocab.map((c, i) => [c, i]));
    }
    // Vocab = the sorted set of unique characters in the corpus. "A token" here is
    // literally one character (no BPE/subwords) — the simplest possible tokenizer.
    // Sorting makes the id↔char mapping deterministic given the same corpus, and
    // the resulting vocab is saved INSIDE the checkpoint so inference is exact.
    static fromText(text) {
        return new CharTokenizer([...new Set(text)].sort());
    }
    get size() {
        return this.vocab.length;
    }
    /** Unknown chars are skipped (terminal corpora are closed-vocabulary anyway). */
    encode(s) {
        const out = new Int32Array(s.length);
        let n = 0;
        for (const ch of s) {
            const id = this.stoi.get(ch);
            if (id !== undefined)
                out[n++] = id;
        }
        return out.subarray(0, n);
    }
    decode(ids) {
        let s = "";
        for (let i = 0; i < ids.length; i++)
            s += this.vocab[ids[i]] ?? "";
        return s;
    }
}

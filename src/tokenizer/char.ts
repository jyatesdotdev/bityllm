// Char-level tokenizer (DESIGN §11): vocab = sorted unique chars of the corpus.
// The vocab travels inside checkpoints so inference reconstructs it exactly.

export interface Tokenizer {
  readonly size: number;
  encode(s: string): Int32Array;
  decode(ids: ArrayLike<number>): string;
}

export class CharTokenizer implements Tokenizer {
  readonly vocab: string[];
  private readonly stoi: Map<string, number>;

  constructor(vocab: string[]) {
    this.vocab = vocab;
    this.stoi = new Map(vocab.map((c, i) => [c, i]));
  }

  static fromText(text: string): CharTokenizer {
    return new CharTokenizer([...new Set(text)].sort());
  }

  get size(): number {
    return this.vocab.length;
  }

  /** Unknown chars are skipped (terminal corpora are closed-vocabulary anyway). */
  encode(s: string): Int32Array {
    const out = new Int32Array(s.length);
    let n = 0;
    for (const ch of s) {
      const id = this.stoi.get(ch);
      if (id !== undefined) out[n++] = id;
    }
    return out.subarray(0, n);
  }

  decode(ids: ArrayLike<number>): string {
    let s = "";
    for (let i = 0; i < ids.length; i++) s += this.vocab[ids[i]] ?? "";
    return s;
  }
}

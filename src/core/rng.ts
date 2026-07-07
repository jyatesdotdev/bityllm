// Seedable PRNG (DESIGN §18): no Math.random anywhere in the library.
// mulberry32 for uniforms, Box–Muller (with spare) for normals.

export class RNG {
  private s: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  /** uniform in [0, 1) */
  random(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** integer in [lo, hi) */
  randint(lo: number, hi: number): number {
    return lo + Math.floor(this.random() * (hi - lo));
  }

  /** standard normal */
  randn(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0, v = 0;
    while (u === 0) u = this.random();
    v = this.random();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
}

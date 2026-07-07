// Module base (DESIGN §8): registers parameters and submodules for recursive
// collection; train(mode) toggles dropout behavior.

import type { Tensor } from "../core/tensor.ts";

export abstract class Module {
  protected _params: Tensor[] = [];
  protected _mods: Module[] = [];
  training = true;

  protected reg<T extends Tensor>(t: T): T {
    this._params.push(t);
    return t;
  }

  protected sub<M extends Module>(m: M): M {
    this._mods.push(m);
    return m;
  }

  parameters(): Tensor[] {
    return [...this._params, ...this._mods.flatMap((m) => m.parameters())];
  }

  train(mode = true): void {
    this.training = mode;
    for (const m of this._mods) m.train(mode);
  }
}

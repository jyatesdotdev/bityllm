// Module base (DESIGN §8): registers parameters and submodules for recursive
// collection; train(mode) toggles dropout behavior.
export class Module {
    _params = [];
    _mods = [];
    training = true;
    reg(t) {
        this._params.push(t);
        return t;
    }
    sub(m) {
        this._mods.push(m);
        return m;
    }
    parameters() {
        return [...this._params, ...this._mods.flatMap((m) => m.parameters())];
    }
    train(mode = true) {
        this.training = mode;
        for (const m of this._mods)
            m.train(mode);
    }
}

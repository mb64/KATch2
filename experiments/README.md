# Experiments

## Installation

```
python3 -m pip install igraph matplotlib

make topozoo
```

## Usage

The following builds an experiment for a Topology Zoo graph:

```
./experiment.py gml/Karen.gml
```

The graph is rendered as `gml/Karen.pdf` (using GraphViz),
and the NetKAT model is emitted as `gml/Karen.nksynth`.

You can run the synthesizer like this:

```
time ../target/release/nksynth --full gml/Karen.nksynth
```

You can also generate an NKPL file that works with the Cornell
OCaml version of the NetKAT solver:
```
./experiment.py --nkpl gml/Karen.gml
```

This emits the model as `gml/Karen.nkpl`.

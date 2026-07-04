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

## Network Model

```
./experiment.py gml/Aarnet.gml
```

The resulting file `gml/Aarnet.nksynth` has the following structure.

```
// NETWORK TOPOLOGY

def topo =
    // LINK: loc=Sydney1, port=1, dir=OUT --> loc:=Canberra2, port:=1, dir:=IN  
    // (loc=1; port=1; out=1; loc:=11; port:=1; out:=0)
    (x[0..5]~1; x[10..13]~1; x[13..14]~1; x[0..5]:=11; x[10..13]:=1; x[13..14]:=0) +
    ...

// FORWARDING POLICY

def pol =
    // RULE: loc=Adelaide1, dst=Adelaide2, dir=IN --> port:=3, dir:=OUT  
    // (loc=14; dst=15; out=0; port:=3; out:=1)
    (x[0..5]~14; x[5..10]~15; x[13..14]~0; x[10..13]:=3; x[13..14]:=1) +

hole h1
def hop = h1; pol; topo
def net = (hop; dup)*

// BLOCK BAD PATHS

// Brisbane1 -> Sydney1 -> Sydney2 -> Melbourne2 -> Adelaide2
// check loc=7; dst=15; net; loc=15 == drop
assert x[0..5]~7; x[5..10]~15; net; x[0..5]~15 = 0

// ALLOW GOOD PATHS

// Melbourne2 -> Adelaide2 -> Alice_Springs
// check (loc=17; port=0; out=0; dst=18; loc:=15; port:=3 ; dup; loc:=18; port:=1 ) <= hop; dup; hop
assert (x[0..5]~17; x[10..13]~0; x[13..14]~0; x[5..10]~18; x[0..5]:=15; x[10..13]:=3 ; dup; x[0..5]:=18; x[10..13]:=1 ) <= hop; dup; hop
```

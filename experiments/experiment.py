#!/usr/bin/env python3

# python3 -m pip install igraph matplotlib

import sys
import igraph as ig
import matplotlib.pyplot as plt
from collections import defaultdict
import subprocess
from pathlib import Path
import re
import random
import argparse

def load_graph(filename):
    """Load a Topology Zoo GML file."""
    return ig.Graph.Read_GML(filename)


def print_summary(g):
    print("=" * 60)
    print("Graph Summary")
    print("=" * 60)

    print(f"Nodes: {g.vcount()}")
    print(f"Edges: {g.ecount()}")

    if g.attributes():
        print("\nGraph attributes:")
        for attr in g.attributes():
            print(f"  {attr}: {g[attr]}")
    else:
        print("\n(No graph attributes)")

    print()


def print_nodes(g):
    print("=" * 60)
    print("Nodes")
    print("=" * 60)

    for v in g.vs:
        print(f"Vertex {v.index}")

        attrs = v.attributes()
        if attrs:
            for k, val in attrs.items():
                print(f"    {k}: {val}")

        print()


def print_edges(g):
    print("=" * 60)
    print("Edges")
    print("=" * 60)

    for i, e in enumerate(g.es):
        src = g.vs[e.source]
        dst = g.vs[e.target]

        src_name = src["label"] if "label" in src.attributes() else src.index
        dst_name = dst["label"] if "label" in dst.attributes() else dst.index

        print(f"Edge {i}: {src_name} <--> {dst_name}")

        attrs = e.attributes()
        if attrs:
            for k, val in attrs.items():
                print(f"    {k}: {val}")

        print()


def geographic_layout(g):
    """Return geographic coordinates if every node has Longitude/Latitude."""
    coords = []

    for v in g.vs:
        attrs = v.attributes()

        if "Longitude" not in attrs or "Latitude" not in attrs:
            return None

        coords.append((
            float(attrs["Longitude"]),
            -float(attrs["Latitude"])   # Flip so north is up
        ))

    return coords


def draw_graph(g):
    layout = geographic_layout(g)

    if layout is None:
        print("Using Kamada-Kawai layout.")
        layout = g.layout("kk")
    else:
        print("Using geographic coordinates.")

    fig, ax = plt.subplots(figsize=(12, 8))

    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_size=10,
        vertex_color="lightblue",
        vertex_frame_color="black",
        vertex_label=g.vs["label"] if "label" in g.vs.attributes() else None,
        vertex_label_size=8,
        edge_width=1,
        margin=40,
    )

    plt.show()

def export_dot(g, filename, engine="neato"):
    """
    Export an igraph Graph to Graphviz DOT format and generate a PDF.

    Parameters
    ----------
    g : igraph.Graph
    filename : str
        Output DOT filename.
    engine : str
        Graphviz layout engine:
            dot    - hierarchical
            neato  - spring model (good default)
            fdp    - force-directed
            sfdp   - scalable force-directed (large graphs)
            circo  - circular
            twopi  - radial
    """

    path = Path(filename)
    dot_path = path.with_suffix(".dot")
    pdf_path = path.with_suffix(".pdf")

    connector = "--" if not g.is_directed() else "->"

    with open(dot_path, "w") as f:
        graph_type = "graph" if not g.is_directed() else "digraph"
        f.write(f"{graph_type} G {{\n")
        f.write("    overlap=false;\n")
        f.write("    splines=true;\n")
        f.write("    node [shape=circle, fontsize=10, width=0.25, fixedsize=true];\n\n")

        # Vertices
        for v in g.vs:
            name = (
                v["label"]
                if "label" in v.attributes()
                else str(v.index)
            )

            # Escape quotes
            name = name.replace('"', '\\"')

            f.write(f'    "{name}";\n')

        f.write("\n")

        # Edges
        for e in g.es:
            u = g.vs[e.source]
            v = g.vs[e.target]

            u_name = (
                u["label"]
                if "label" in u.attributes()
                else str(u.index)
            ).replace('"', '\\"')

            v_name = (
                v["label"]
                if "label" in v.attributes()
                else str(v.index)
            ).replace('"', '\\"')

            f.write(f'    "{u_name}" {connector} "{v_name}";\n')

        f.write("}\n")

    subprocess.run(
        [
            engine,
            "-Tpdf",
            str(dot_path),
            "-o",
            str(pdf_path),
        ],
        check=True,
    )

    print(f"Wrote {dot_path}")
    print(f"Wrote {pdf_path}")


def write_netkat(g, filename, fmt_katch):
    """
    Generate a NetKAT model from an igraph topology.

    Assumptions:
      - Every vertex has a unique 'label' attribute.
      - Routing policy is shortest-path forwarding.
      - Packets carry fields:
            loc   -- current switch
            dst   -- destination switch
            port  -- output/input port
    """

    inline_consts = True
    expand_indices = False # requires inline_consts=True
    suppress_nonempty = True
    num_bad_paths = 2
    num_good_paths = 3
    rand_seed = 3

    random.seed(rand_seed)

    path = Path(filename)
    netkat_path = path.with_suffix(".nksynth" if fmt_katch else ".nkpl")

    def nk_name(label: str) -> str:
        """
        Convert a topology node label into a valid NetKAT identifier.
        """
        # Replace whitespace with underscores
        label = re.sub(r"\s+", "_", label)

        # Replace any remaining non-identifier characters
        label = re.sub(r"[^A-Za-z0-9_]", "_", label)

        # Identifiers cannot begin with a digit
        if label and label[0].isdigit():
            label = "_" + label

        return label

    # ------------------------------------------------------------------
    # Assign port numbers
    # ------------------------------------------------------------------

    port_of = {}
    next_port = defaultdict(lambda: 1)

    for e in g.es:
        u = e.source
        v = e.target

        if (u, v) not in port_of:
            port_of[(u, v)] = next_port[u]
            next_port[u] += 1

        if (v, u) not in port_of:
            port_of[(v, u)] = next_port[v]
            next_port[v] += 1

    max_next_port = max(next_port.values(), default=1)-1
    port_bits = max_next_port.bit_length()
    #print(f"max = {max_next_port}, bits = {port_bits}")

    #labels = g.vs["label"]
    labels = [nk_name(v["label"]) for v in g.vs]

    with open(netkat_path, "w") as f:

        ############################################################
        # Constants
        ############################################################

        label_table = {}
        for i, label in enumerate(labels, start=1):
            if not inline_consts:
                f.write(f"{label} = {i}\n")
            label_table[label] = i

        if not inline_consts:
            f.write("\n")

        num_elements = len(label_table)
        loc_bits = num_elements.bit_length()
        #print(f"num elements = {num_elements}, bits = {loc_bits}")

        def bits_field(f):
            match f:
                case "loc":
                    return (0,loc_bits)
                case "dst":
                    return (loc_bits,loc_bits+loc_bits)
                case "port":
                    return (loc_bits+loc_bits,loc_bits+loc_bits+port_bits)
                case "out":
                    return (loc_bits+loc_bits+port_bits,loc_bits+loc_bits+port_bits+1)

        def str_field_ops(f, v, op1, op2):
            if fmt_katch:
                bits = bits_field(f)
                if expand_indices:
                    width = bits[1]-bits[0]
                    binary = f"{int(v):0{width}b}"
                    pairs = list(zip(range(bits[0], bits[1]), map(int, binary)))
                    s = "; ".join(f"x{i}{op2}{bit}" for i, bit in pairs)
                else:
                    s = f"x[{bits[0]}..{bits[1]}]{op1}{v}"
                #print(f"{s} --> {pairs}")
                return s
            else:
                return f"{f}{op2}{v}"

        def str_field_test(f, v):
            return str_field_ops(f, v, "~", "=")

        def str_field_assign(f, v):
            return str_field_ops(f, v, ":=", ":=")

        def str_label(u):
            u_label = labels[u]
            if inline_consts:
                u_label = str(label_table[u_label])
            return u_label

        ############################################################
        # topo
        ############################################################

        f.write(f"{'def ' if fmt_katch else ''}topo =\n")

        first = True

        for e in g.es:
            u, v = e.tuple

            pu = port_of[(u, v)]
            pv = port_of[(v, u)]

            for (src, src_port, dst, dst_port) in [
                (u, pu, v, pv),
                (v, pv, u, pu),
            ]:
                if not first:
                    f.write("\n +\n")
                first = False

                src_label = str_label(src)
                dst_label = str_label(dst)

                f.write(
                    "    "
                    f"({str_field_test('loc',src_label)}; "
                    f"{str_field_test('port',src_port)}; "
                    f"{str_field_test('out',1)}; "
                    f"{str_field_assign('loc',dst_label)}; "
                    f"{str_field_assign('port',dst_port)}; "
                    f"{str_field_assign('out',0)})"
                )

        f.write("\n\n")

        ############################################################
        # pol
        ############################################################

        f.write(f"{'def ' if fmt_katch else ''}pol =\n")

        flood = False
        if flood:
            first = True

            for here in range(g.vcount()):

                #
                # Every port on this switch.
                #
                ports = set()

                for nbr in g.neighbors(here):
                    ports.add(port_of[(here, nbr)])   # output ports
                    ports.add(port_of[(nbr, here)])   # input ports (same numbers if undirected)

                #
                # Flood every inbound packet to every outbound port.
                #
                for in_port in sorted(ports | {0}):

                    for out_port in sorted(ports):

                        here_label = str_label(here)

                        rule = (
                            f"({str_field_test('loc',here_label)}; "
                            f"{str_field_test('port',in_port)}; "
                            f"{str_field_test('out',0)}; "
                            f"{str_field_assign('port',out_port)}; "
                            f"{str_field_assign('out',1)})"
                        )

                        if not first:
                            f.write("\n +\n")
                        first = False

                        f.write("    " + rule)

            f.write("\n\n")
        else:
            #        Shortest paths:
            #        (sw1, sw2) -->  sw1, sw3, sw2
            #        (sw1, sw3) -->  sw1, sw3
            #        (sw1, sw4) -->  sw1, sw3, sw2, sw4
            #
            #        Next-switch function of the following type:
            #        (src, dst) --> next
            #
            #        Here is the function defined for the above shortest paths
            #        (sw1, sw2) --> sw
            #        (sw1, sw3) --> sw
            #        (sw1, sw4) --> sw

            ############################################################
            # pol
            ############################################################

            #
            # Compute forwarding table:
            #
            #     next_hop[(here, dst)] = next_switch
            #
            # from all-pairs shortest paths.
            #

            next_hop = {}
            paths = []

            for src in range(g.vcount()):
                for dst in range(g.vcount()):

                    if src == dst:
                        continue

                    path = g.get_shortest_paths(src, to=dst, output="vpath")[0]
                    paths.append(path)

                    if len(path) < 2:
                        continue

                    #
                    # Record the next hop for every switch on the path.
                    #
                    # Example:
                    #
                    #   A -> B -> C -> D
                    #
                    # produces
                    #
                    #   next(A,D)=B
                    #   next(B,D)=C
                    #   next(C,D)=D
                    #

                    for i in range(len(path) - 1):
                        here = path[i]
                        nxt = path[i + 1]

                        #
                        # Only keep the first rule we discover.
                        # Every shortest path should induce the same next hop
                        # for a given (here,dst).
                        #

                        next_hop.setdefault((here, dst), nxt)

            ############################################################
            # Emit forwarding policy
            ############################################################

            first = True

            for (here, dst), nxt in sorted(
                next_hop.items(),
                key=lambda item: (
                    labels[item[0][0]],   # current switch
                    labels[item[0][1]],   # destination
                ),
            ):

                out_port = port_of[(here, nxt)]

                here_label = str_label(here)
                dst_label = str_label(dst)

                rule = (
                    f"({str_field_test('loc',here_label)}; "
                    f"{str_field_test('out',0)}; "
                    f"{str_field_test('dst',dst_label)}; "
                    f"{str_field_assign('port',out_port)}; "
                    f"{str_field_assign('out',1)})"
                )

                if not first:
                    f.write("\n +\n")

                first = False
                f.write("    " + rule)

            f.write("\n\n")



        ############################################################

        f.write(f"{'hole h1' if fmt_katch else 'hole = skip'}\n")
        f.write(f"{'def ' if fmt_katch else ''}hop = {'h1' if fmt_katch else 'hole'}; pol; topo\n")
        f.write(f"{'def ' if fmt_katch else ''}net = (hop; dup)*\n\n")

        ############################################################
        # Reachability checks
        ############################################################

        if not suppress_nonempty:
            for s in range(g.vcount()):
                for d in range(g.vcount()):
                    if s == d:
                        continue

                    s_label = str_label(s)
                    d_label = str_label(d)

                    f.write(
                        f"{'assert' if fmt_katch else 'check'} {str_field_test('loc',s_label)}; "
                        f"net; "
                        f"{str_field_test('loc',d_label)} {'!=' if fmt_katch else '!=='} drop\n"
                    )

            f.write("\n")

        ############################################################
        # Paths
        ############################################################

        bad_paths = random.sample(paths, k=num_bad_paths)

        for p in bad_paths:
            name = " -> ".join(labels[v] for v in p)
            paths.remove(p)
            s_label = str_label(p[0])
            d_label = str_label(p[-1])
            #f.write(f"bad path = {name}, first={s_label}, last={d_label}\n")

            f.write(
                f"{'assert' if fmt_katch else 'check'} {str_field_test('loc',s_label)}; "
                f"{str_field_test('dst',d_label)}; net; "
                f"{str_field_test('loc',d_label)} {'=' if fmt_katch else '=='} {'0' if fmt_katch else 'drop'}\n"
            )

        f.write("\n")

        good_paths = random.sample(paths, min(num_good_paths, len(paths)))

        for p in good_paths:
            name = " -> ".join(labels[v] for v in p)
            if True or name=="POR -> AVL -> WLG -> NLS":
                #print("\n\nPath:", name)
                #print("Ports: ", port_of)
                items = []
                first = ""
                prev = None
                for x in p:
                    #print("Item: ", labels[x])

                    x_label = str_label(x)

                    if prev is None:
                        first = (
                            f"{str_field_test('loc',x_label)}; "
                            f"{str_field_test('port',0)}; "
                            f"{str_field_test('out',0)}; "
                        )
                    else:
                        #print("  port: ", str(port_of[(x,prev)]))
                        items.append(
                            f"{str_field_assign('loc',x_label)}; "
                            f"{str_field_assign('port',port_of[(x,prev)])} "
                        )
                    prev = x
                first += f"{str_field_test('dst',x_label)}; "
                hops = ["hop"] * len(items)
                #print("First: ", first)
                #print("Items: ", "; dup; ".join(items))
                #print("Hops: ", "; dup; ".join(hops))

                f.write(
                    f"{'assert' if fmt_katch else 'check'} ({first}{'; dup; '.join(items)})"
                    f" <= {'; dup; '.join(hops)}\n"
                )
    print(f"Wrote {netkat_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nkpl", action="store_true")
    parser.add_argument("filenames", nargs="+")

    args = parser.parse_args()

    print(args.nkpl)
    print(args.filenames)

    for filename in args.filenames:
        g = load_graph(filename)

        print_summary(g)
        print_nodes(g)
        print_edges(g)

        export_dot(g, filename, engine="neato")
        write_netkat(g, filename, not args.nkpl)

        #draw_graph(g)


if __name__ == "__main__":
    main()

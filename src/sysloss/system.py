# MIT License
#
# Copyright (c) 2024, Geir Drange
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rustworkx as rx
import numpy as np
from rich.tree import Tree
import json
import pandas as pd
from sysloss.components import *
from sysloss.components import (
    ComponentTypes,
    _get_opt,
    _get_mand,
    _get_eff,
    RS_DEFAULT,
    LIMITS_DEFAULT,
)


class System:
    """System to be analyzed."""

    def __init__(self, name: str, source):
        self._g = None
        if not isinstance(source, Source):
            raise ValueError("Error: First component of system must be a source!")

        self._g = rx.PyDAG(check_cycle=True, multigraph=False, attrs={})
        pidx = self._g.add_node(source)
        self._g.attrs["name"] = name
        self._g.attrs["nodes"] = {}
        self._g.attrs["nodes"][source.params["name"]] = pidx

    @classmethod
    def from_file(cls, fname: str):
        """Load system from json file"""
        with open(fname, "r") as f:
            sys = json.load(f)

        entires = list(sys.keys())
        sysname = _get_mand(sys, "name")
        ver = _get_mand(sys, "version")
        # add sources
        for e in range(2, len(entires)):
            vo = _get_mand(sys[entires[e]]["params"], "vo")
            rs = _get_opt(sys[entires[e]]["params"], "rs", RS_DEFAULT)
            lim = _get_opt(sys[entires[e]], "limits", LIMITS_DEFAULT)
            if e == 2:
                self = cls(sysname, Source(entires[e], vo=vo, rs=rs, limits=lim))
            else:
                self.add_source(Source(entires[e], vo=vo, rs=rs, limits=lim))
            # add childs
            if sys[entires[e]]["childs"] != {}:
                for p in list(sys[entires[e]]["childs"].keys()):
                    for c in sys[entires[e]]["childs"][p]:
                        cname = _get_mand(c["params"], "name")
                        # print("  " + cname)
                        limits = _get_opt(c, "limits", LIMITS_DEFAULT)
                        iq = _get_opt(c["params"], "iq", 0.0)
                        if c["type"] == "CONVERTER":
                            vo = _get_mand(c["params"], "vo")
                            eff = _get_mand(c["params"], "eff")
                            self.add_comp(
                                p,
                                comp=Converter(
                                    cname, vo=vo, eff=eff, iq=iq, limits=limits
                                ),
                            )
                        elif c["type"] == "LINREG":
                            vo = _get_mand(c["params"], "vo")
                            vdrop = _get_opt(c["params"], "vdrop", 0.0)
                            self.add_comp(
                                p,
                                comp=LinReg(
                                    cname, vo=vo, vdrop=vdrop, iq=iq, limits=limits
                                ),
                            )
                        elif c["type"] == "LOSS":
                            rs = _get_mand(c["params"], "rs")
                            vdrop = _get_mand(c["params"], "vdrop")
                            self.add_comp(
                                p, comp=Loss(cname, rs=rs, vdrop=vdrop, limits=limits)
                            )
                        elif c["type"] == "LOAD":
                            if "pwr" in c["params"]:
                                pwr = _get_mand(c["params"], "pwr")
                                self.add_comp(
                                    p, comp=PLoad(cname, pwr=pwr, limits=limits)
                                )
                            elif "rs" in c["params"]:
                                rs = _get_mand(c["params"], "rs")
                                self.add_comp(
                                    p, comp=RLoad(cname, rs=rs, limits=limits)
                                )
                            else:
                                ii = _get_mand(c["params"], "ii")
                                self.add_comp(
                                    p, comp=ILoad(cname, ii=ii, limits=limits)
                                )

        return self

    def _get_index(self, name: str):
        """Get node index from component name"""
        if name in self._g.attrs["nodes"]:
            return self._g.attrs["nodes"][name]

        return -1

    def _chk_parent(self, parent: str):
        """Check if parent exists"""
        if not parent in self._g.attrs["nodes"].keys():
            raise ValueError('Error: Parent name "{}" not found!'.format(parent))

        return True

    def _chk_name(self, name: str):
        """Check if component name is valid"""
        # check if name exists
        if name in self._g.attrs["nodes"].keys():
            raise ValueError('Error: Component name "{}" is already used!'.format(name))

        return True

    def _get_childs_tree(self, node):
        """Get dict of parent/childs"""
        childs = list(rx.bfs_successors(self._g, node))
        cdict = {}
        for c in childs:
            cs = []
            for l in c[1]:
                cs += [self._g.attrs["nodes"][l.params["name"]]]
            cdict[self._g.attrs["nodes"][c[0].params["name"]]] = cs
        return cdict

    def _get_nodes(self):
        """Get list of nodes in system"""
        return [n for n in self._g.node_indices()]

    def _get_childs(self):
        """Get list of children of each node"""
        nodes = self._get_nodes()
        cs = list(-np.ones(max(nodes) + 1, dtype=np.int32))
        for n in nodes:
            if self._g.out_degree(n) > 0:
                ind = [i for i in self._g.successor_indices(n)]
                cs[n] = ind
        return cs

    def _get_parents(self):
        """Get list of parent of each node"""
        nodes = self._get_nodes()
        ps = list(-np.ones(max(nodes) + 1, dtype=np.int32))
        for n in nodes:
            if self._g.in_degree(n) > 0:
                ind = [i for i in self._g.predecessor_indices(n)]
                ps[n] = ind
        return ps

    def _get_sources(self):
        """Get list of sources"""
        tn = [n for n in rx.topological_sort(self._g)]
        return [n for n in tn if isinstance(self._g[n], Source)]

    def _get_topo_sort(self):
        """Get nodes topological sorted"""
        tps = rx.topological_sort(self._g)
        return [n for n in tps]

    def _sys_vars(self):
        """Get system variable lists"""
        vn = max(self._get_nodes()) + 1  # highest node index + 1
        v = list(np.zeros(vn))  # voltages
        i = list(np.zeros(vn))  # currents
        return v, i

    def _make_rtree(self, adj, node):
        """Create Rich tree"""
        tree = Tree(node)
        for child in adj.get(node, []):
            tree.add(self._make_rtree(adj, child))
        return tree

    def add_comp(self, parent: str, *, comp):
        """Add component to system"""
        # check that parent exists
        self._chk_parent(parent)
        # check that component name is unique
        self._chk_name(comp.params["name"])
        # check that parent allows component type as child
        pidx = self._get_index(parent)
        if not comp.component_type in self._g[pidx].child_types:
            raise ValueError(
                "Error: Parent does not allow child of type {}!".format(
                    comp.component_type.name
                )
            )
        cidx = self._g.add_child(pidx, comp, None)
        self._g.attrs["nodes"][comp.params["name"]] = cidx

    def add_source(self, source):
        """Add new source"""
        self._chk_name(source.params["name"])
        if not isinstance(source, Source):
            raise ValueError("Error: Component must be a source!")

        pidx = self._g.add_node(source)
        self._g.attrs["nodes"][source.params["name"]] = pidx

    def change_comp(self, name: str, *, comp):
        """Replace component with a new one"""
        # if component name changes, check that it is unique
        if name != comp.params["name"]:
            self._chk_name(comp.params["name"])

        # source can only be changed to source
        eidx = self._get_index(name)
        if self._g[eidx].component_type == ComponentTypes.SOURCE:
            if not isinstance(comp, Source):
                raise ValueError("Error: Source cannot be changed to other type!")

        # check that parent allows component type as child
        parents = self._get_parents()
        if parents[eidx] != -1:
            if not comp.component_type in self._g[parents[eidx][0]].child_types:
                raise ValueError(
                    "Error: Parent does not allow child of type {}!".format(
                        comp.component_type.name
                    )
                )
        self._g[eidx] = comp
        # replace node name in graph dict
        del [self._g.attrs["nodes"][name]]
        self._g.attrs["nodes"][comp.params["name"]] = eidx

    def del_comp(self, name: str, *, del_childs: bool = True):
        eidx = self._get_index(name)
        if eidx == -1:
            raise ValueError("Error: Component name does not exist!")
        parents = self._get_parents()
        if parents[eidx] == -1:  # source node
            if not del_childs:
                raise ValueError("Error: Source must be deleted with its childs")
            if len(self._get_sources()) < 2:
                raise ValueError("Error: Cannot delete the last source node!")
        childs = self._get_childs()
        # if not leaf, check if child type is allowed by parent type (not possible?)
        # if leaves[eidx] == 0:
        #     for c in childs[eidx]:
        #         if not self._g[c].component_type in self._g[parents[eidx]].child_types:
        #             raise ValueError(
        #                 "Error: Parent and child of component are not compatible!"
        #             )
        # delete childs first if selected
        if del_childs:
            for c in rx.descendants(self._g, eidx):
                del [self._g.attrs["nodes"][self._g[c].params["name"]]]
                self._g.remove_node(c)
        # delete node
        self._g.remove_node(eidx)
        del [self._g.attrs["nodes"][name]]
        # restore links between new parent and childs, unless deleted
        if not del_childs:
            if childs[eidx] != -1:
                for c in childs[eidx]:
                    self._g.add_edge(parents[eidx][0], c, None)

    def tree(self, name=""):
        """Print tree structure, starting from node name"""
        if not name == "":
            if not name in self._g.attrs["nodes"].keys():
                raise ValueError("Error: Component name is not valid!")
            root = [name]
        else:
            ridx = self._get_sources()
            root = [self._g[n].params["name"] for n in ridx]

        t = Tree(self._g.attrs["name"])
        for n in root:
            adj = rx.bfs_successors(self._g, self._g.attrs["nodes"][n])
            ndict = {}
            for i in adj:
                c = []
                for j in i[1]:
                    c += [j.params["name"]]
                ndict[i[0].params["name"]] = c
            t.add(self._make_rtree(ndict, n))
        return t

    def _sys_init(self):
        """Create vectors of init values for solver"""
        v, i = self._sys_vars()
        for n in self._get_nodes():
            v[n] = self._g[n]._get_outp_voltage()
            i[n] = self._g[n]._get_inp_current()
        return v, i

    def _fwd_prop(self, v: float, i: float):
        """Forward propagation of voltages"""
        vo, _ = self._sys_vars()
        # update output voltages (per node)
        for n in self._topo_nodes:
            p = self._parents[n]
            if self._childs[n] == -1:  # leaf
                if p == -1:  # root
                    vo[n] = self._g[n]._solv_outp_volt(0.0, 0.0, 0.0)
                else:
                    vo[n] = self._g[n]._solv_outp_volt(v[p[0]], i[n], 0.0)
            else:
                # add currents into childs
                isum = 0
                for c in self._childs[n]:
                    isum += i[c]
                if p == -1:  # root
                    vo[n] = self._g[n]._solv_outp_volt(0.0, 0.0, isum)
                else:
                    vo[n] = self._g[n]._solv_outp_volt(v[p[0]], i[n], isum)
        return vo

    def _back_prop(self, v: float, i: float):
        """Backward propagation of currents"""
        _, ii = self._sys_vars()
        # update input currents (per node)
        for n in self._topo_nodes[::-1]:
            p = self._parents[n]
            if self._childs[n] == -1:  # leaf
                if p == -1:  # root
                    ii[n] = self._g[n]._solv_inp_curr(v[n], 0.0, 0.0)
                else:
                    ii[n] = self._g[n]._solv_inp_curr(v[p[0]], 0.0, 0.0)
            else:
                isum = 0.0
                for c in self._childs[n]:
                    isum += i[c]
                if p == -1:  # root
                    ii[n] = self._g[n]._solv_inp_curr(v[n], v[n], isum)
                else:
                    ii[n] = self._g[n]._solv_inp_curr(v[p[0]], v[n], isum)

        return ii

    def _rel_update(self):
        """Update lists with component relationships"""
        self._parents = self._get_parents()
        self._childs = self._get_childs()
        self._topo_nodes = self._get_topo_sort()

    def _get_parent_name(self, node):
        """Get parent name of node"""
        if self._parents[node] == -1:
            return ""
        return self._g[self._parents[node][0]].params["name"]

    def _solve(self, vtol=1e-5, itol=1e-6, maxiter=1000, quiet=True):
        """Solver"""
        v, i = self._sys_init()
        iters = 0
        while iters <= maxiter:
            vi = self._fwd_prop(v, i)
            ii = self._back_prop(vi, i)
            iters += 1
            if np.allclose(np.array(v), np.array(vi), rtol=vtol) and np.allclose(
                np.array(i), np.array(ii), rtol=itol
            ):
                if not quiet:
                    print("Tolerances met after {} iterations".format(iters))
                break
            v, i = vi, ii
        return v, i, iters

    def solve(self, *, vtol=1e-5, itol=1e-6, maxiter=1000, quiet=True):
        """Analyze system"""
        self._rel_update()
        # solve
        v, i, iters = self._solve(vtol, itol, maxiter, quiet)
        if iters > maxiter:
            print("Analysis aborted after {} iterations".format(iters - 1))
            return None

        # calculate results for each node
        names, parent, typ, pwr, loss = [], [], [], [], []
        eff, warn, vsi, iso, vso, isi = [], [], [], [], [], []
        domain, dname = [], "none"
        sources, dwarns = {}, {}
        for n in self._topo_nodes:  # [vi, vo, ii, io]
            names += [self._g[n].params["name"]]
            if self._g[n].component_type.name == "SOURCE":
                dname = self._g[n].params["name"]
            domain += [dname]
            vi = v[n]
            vo = v[n]
            ii = i[n]
            io = i[n]
            p = self._parents[n]

            if p == -1:  # root
                vi = v[n] + self._g[n].params["rs"] * ii
            elif self._childs[n] == -1:  # leaf
                vi = v[p[0]]
                io = 0.0
            else:
                io = 0.0
                for c in self._childs[n]:
                    io += i[c]
                vi = v[p[0]]
            parent += [self._get_parent_name(n)]
            p, l, e = self._g[n]._solv_pwr_loss(vi, vo, ii, io)
            pwr += [p]
            loss += [l]
            eff += [e]
            typ += [self._g[n].component_type.name]
            if self._g[n].component_type.name == "SOURCE":
                sources[dname] = vi
                dwarns[dname] = 0
            w = self._g[n]._solv_get_warns(vi, vo, ii, io)
            warn += [w]
            if w != "":
                dwarns[dname] = 1
            vsi += [vi]
            iso += [io]
            vso += [v[n]]
            isi += [i[n]]

        # subsystems summary
        for d in range(len(sources)):
            names += ["Subsystem {}".format(list(sources.keys())[d])]
            typ += [""]
            parent += [""]
            domain += [""]
            vsi += [sources[list(sources.keys())[d]]]
            vso += [""]
            isi += [""]
            iso += [""]
            pwr += [""]
            loss += [""]
            eff += [""]
            if dwarns[list(sources.keys())[d]] > 0:
                warn += ["Yes"]
            else:
                warn += [""]

        # system total
        names += ["System total"]
        typ += [""]
        parent += [""]
        domain += [""]
        vsi += [""]
        vso += [""]
        isi += [""]
        iso += [""]
        pwr += [""]
        loss += [""]
        eff += [""]
        if any(warn):
            warn += ["Yes"]
        else:
            warn += [""]

        # report
        res = {}
        res["Component"] = names
        res["Type"] = typ
        res["Parent"] = parent
        res["Domain"] = domain
        res["Vin (V)"] = vsi
        res["Vout (V)"] = vso
        res["Iin (A)"] = isi
        res["Iout (A)"] = iso
        res["Power (W)"] = pwr
        res["Loss (W)"] = loss
        res["Efficiency (%)"] = eff
        res["Warnings"] = warn
        df = pd.DataFrame(res)

        # update subsystem power/loss/efficiency
        for d in range(len(sources)):
            src = list(sources.keys())[d]
            idx = df[df.Component == "Subsystem {}".format(src)].index[0]
            pwr = df[(df.Domain == src) & (df.Type == "SOURCE")]["Power (W)"].values[0]
            df.at[idx, "Power (W)"] = pwr
            loss = df[df.Domain == src]["Loss (W)"].sum()
            df.at[idx, "Loss (W)"] = loss
            df.at[idx, "Efficiency (%)"] = _get_eff(pwr, loss)

        # update system total
        pwr = df[(df.Domain == "") & (df["Power (W)"] != "")]["Power (W)"].sum()
        idx = df.index[-1]
        df.at[idx, "Power (W)"] = pwr
        loss = df[(df.Domain == "") & (df["Loss (W)"] != "")]["Loss (W)"].sum()
        df.at[idx, "Loss (W)"] = loss
        df.at[idx, "Efficiency (%)"] = _get_eff(pwr, loss)

        # if only one subsystem, delete subsystem row
        if len(sources) < 2:
            df.drop(len(df) - 2, inplace=True)
            df.reset_index(inplace=True, drop=True)
        return df

    def params(self, limits=False):
        """Return component parameters"""
        self._rel_update()
        names, typ, parent, vo, vdrop = [], [], [], [], []
        iq, rs, eff, ii, pwr = [], [], [], [], []
        lii, lio, lvi, lvo = [], [], [], []
        domain, dname = [], "none"
        for n in self._topo_nodes:
            names += [self._g[n].params["name"]]
            typ += [self._g[n].component_type.name]
            if self._g[n].component_type.name == "SOURCE":
                dname = self._g[n].params["name"]
            domain += [dname]
            _vo, _vdrop, _iq, _rs, _eff, _ii, _pwr = "", "", "", "", "", "", ""
            if self._g[n].component_type == ComponentTypes.SOURCE:
                _vo = self._g[n].params["vo"]
                _rs = self._g[n].params["rs"]
            elif self._g[n].component_type == ComponentTypes.LOAD:
                if "pwr" in self._g[n].params:
                    _pwr = self._g[n].params["pwr"]
                elif "rs" in self._g[n].params:
                    _rs = self._g[n].params["rs"]
                else:
                    _ii = self._g[n].params["ii"]
            elif self._g[n].component_type == ComponentTypes.CONVERTER:
                _vo = self._g[n].params["vo"]
                _iq = self._g[n].params["iq"]
                _eff = self._g[n].params["eff"]
            elif self._g[n].component_type == ComponentTypes.LINREG:
                _vo = self._g[n].params["vo"]
                _vdrop = self._g[n].params["vdrop"]
                _iq = self._g[n].params["iq"]
            elif self._g[n].component_type == ComponentTypes.LOSS:
                _vdrop = self._g[n].params["vdrop"]
                _rs = self._g[n].params["rs"]
            vo += [_vo]
            vdrop += [_vdrop]
            iq += [_iq]
            rs += [_rs]
            eff += [_eff]
            ii += [_ii]
            pwr += [_pwr]
            parent += [self._get_parent_name(n)]
            if limits:
                lii += [_get_opt(self._g[n].limits, "ii", LIMITS_DEFAULT["ii"])]
                lio += [_get_opt(self._g[n].limits, "io", LIMITS_DEFAULT["io"])]
                lvi += [_get_opt(self._g[n].limits, "vi", LIMITS_DEFAULT["vi"])]
                lvo += [_get_opt(self._g[n].limits, "vo", LIMITS_DEFAULT["vo"])]
        # report
        res = {}
        res["Component"] = names
        res["Type"] = typ
        res["Parent"] = parent
        res["Domain"] = domain
        res["vo (V)"] = vo
        res["vdrop (V)"] = vdrop
        res["iq (A)"] = iq
        res["rs (Ohm)"] = rs
        res["eff (%)"] = eff
        res["ii (A)"] = ii
        res["pwr (W)"] = pwr
        if limits:
            res["vi limits (V)"] = lvi
            res["vo limits (V)"] = lvo
            res["ii limits (A)"] = lii
            res["io limits (A)"] = lio
        return pd.DataFrame(res)

    def save(self, fname, *, indent=4):
        """Save system as json file"""
        self._rel_update()
        sys = {"name": self._g.attrs["name"], "version": "0.10.0"}  # TODO: version
        ridx = self._get_sources()
        root = [self._g[n].params["name"] for n in ridx]
        for r in range(len(ridx)):
            tree = self._get_childs_tree(ridx[r])
            cdict = {}
            if tree != {}:
                for e in tree:
                    childs = []
                    for c in tree[e]:
                        childs += [
                            {
                                "type": self._g[c].component_type.name,
                                "params": self._g[c].params,
                                "limits": self._g[c].limits,
                            }
                        ]
                    cdict[self._g[e].params["name"]] = childs
            sys[root[r]] = {
                "type": self._g[ridx[r]].component_type.name,
                "params": self._g[ridx[r]].params,
                "limits": self._g[ridx[r]].limits,
                "childs": cdict,
            }

        with open(fname, "w") as f:
            json.dump(sys, f, indent=indent)

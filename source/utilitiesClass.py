# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         utilitiesClass.py
# Description:
# ===========================================================================

import os
import time
import networkx as nx
import matplotlib
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import PatchCollection, CircleCollection


matplotlib.use("Qt4Agg")
import numpy as np
import bisect

TOL = 1e-8


class Utilities:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def get_time():
        return time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime())

    @staticmethod
    def get_time_for_log():
        return time.strftime("%H:%M:%S", time.localtime())

    @staticmethod
    def is_eq_tol(a, b, tol=TOL):
        return abs(a - b) <= tol

    @staticmethod
    def is_not_eq_tol(a, b, tol=TOL):
        return abs(a - b) > tol

    @staticmethod
    def is_geq_tol(a, b, tol=TOL):
        return a - b + tol >= 0

    @staticmethod
    def get_insertion_point_left(L, el):
        return bisect.bisect_left(L, el)



    @staticmethod
    def get_shortest_path_network(network, labels=None):

        if not labels:
            # Use transit-times as edge weight
            labels = nx.single_source_dijkstra_path_length(G=network, source='s',
                                                           weight='transitTime')  # Compute node distance from source

        # Create shortest path network containing _all_ shortest paths
        # shortestPathEdges = [(edge[0], edge[1]) for edge in network.edges() if labels[edge[0]] + network[edge[0]][edge[1]]['transitTime'] <= labels[edge[1]]]

        shortestPathEdges = [(edge[0], edge[1]) for edge in network.edges() if Utilities.is_geq_tol(labels[edge[1]],
                                                                                                    labels[edge[0]] +
                                                                                                    network[edge[0]][
                                                                                                        edge[1]][
                                                                                                        'transitTime'])]
        shortestPathNetwork = nx.DiGraph()
        shortestPathNetwork.add_nodes_from(network)
        shortestPathNetwork.add_edges_from(shortestPathEdges)

        for edge in shortestPathEdges:
            v, w = edge[0], edge[1]
            shortestPathNetwork[v][w]['capacity'] = network[v][w]['capacity']
            shortestPathNetwork[v][w]['transitTime'] = network[v][w]['transitTime']

        for w in shortestPathNetwork:
            shortestPathNetwork.node[w]['dist'] = labels[w]
            shortestPathNetwork.node[w]['label'] = network.node[w]['label']
            shortestPathNetwork.node[w]['position'] = network.node[w]['position']

        return shortestPathNetwork

    @staticmethod
    def compute_min_capacity(network):
        minimumCapacity = float('inf')
        for edge in network.edges():
            v, w = edge[0], edge[1]
            minimumCapacity = min([minimumCapacity, network[v][w]['capacity']])

        return minimumCapacity

    @staticmethod
    def join_intersect_dicts(dict1, dict2):
        return {key: (dict1[key], dict2[key]) for key in dict1 if key in dict2}

    @staticmethod
    def add_and_round_up(x, n):
        x += n
        return x if x % 10 == 0 else x + 10 - x % 10

    @staticmethod
    def round_up(x):
        return x if x % 10 == 0 else x + 10 - x % 10

    @staticmethod
    def draw_edges(G, pos,
                   edgelist=None,
                   width=1.0,
                   edge_color='k',
                   style='solid',
                   alpha=1.0,
                   edge_cmap=None,
                   edge_vmin=None,
                   edge_vmax=None,
                   ax=None,
                   arrows=True,
                   label=None,
                   **kwds):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.cbook as cb
            from matplotlib.colors import colorConverter, Colormap
            from matplotlib.collections import LineCollection
            import numpy
        except ImportError:
            raise ImportError("Matplotlib required for draw()")
        except RuntimeError:
            print("Matplotlib unable to open display")
            raise

        if ax is None:
            ax = plt.gca()

        if edgelist is None:
            edgelist = G.edges()

        if not edgelist or len(edgelist) == 0:  # no edges!
            return None

        # set edge positions
        edge_pos = numpy.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

        if not cb.iterable(width):
            lw = (width,)
        else:
            lw = width

        if not cb.is_string_like(edge_color) \
                and cb.iterable(edge_color) \
                and len(edge_color) == len(edge_pos):
            if numpy.alltrue([cb.is_string_like(c)
                              for c in edge_color]):
                # (should check ALL elements)
                # list of color letters such as ['k','r','k',...]
                edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                     for c in edge_color])
            elif numpy.alltrue([not cb.is_string_like(c)
                                for c in edge_color]):
                # If color specs are given as (rgb) or (rgba) tuples, we're OK
                if numpy.alltrue([cb.iterable(c) and len(c) in (3, 4)
                                  for c in edge_color]):
                    edge_colors = tuple(edge_color)
                else:
                    # numbers (which are going to be mapped with a colormap)
                    edge_colors = None
            else:
                raise ValueError('edge_color must consist of either color names or numbers')
        else:
            if cb.is_string_like(edge_color) or len(edge_color) == 1:
                edge_colors = (colorConverter.to_rgba(edge_color, alpha),)
            else:
                raise ValueError(
                    'edge_color must be a single color or list of exactly m colors where m is the number or edges')

        edgeCollection = nx.draw_networkx_edges(G, pos,
                                                edgelist,
                                                width,
                                                edge_color,
                                                style,
                                                alpha,
                                                edge_cmap,
                                                edge_vmin,
                                                edge_vmax,
                                                ax,
                                                arrows=False,
                                                label=label)

        if G.is_directed() and arrows:

            # a directed graph hack
            # draw thick line segments at head end of edge
            # waiting for someone else to implement arrows that will work
            arrow_colors = edge_colors
            a_pos = []
            p = 1.0 - 0.25  # make head segment 25 percent of edge length
            for src, dst in edge_pos:
                x1, y1 = src
                x2, y2 = dst
                dx = x2 - x1  # x offset
                dy = y2 - y1  # y offset
                d = np.sqrt(float(dx ** 2 + dy ** 2))  # length of edge
                if d == 0:  # source and target at same position
                    continue
                if dx == 0:  # vertical edge
                    xa = x2
                    ya = dy * p + y1
                if dy == 0:  # horizontal edge
                    ya = y2
                    xa = dx * p + x1
                else:
                    theta = numpy.arctan2(dy, dx)
                    xa = p * d * numpy.cos(theta) + x1
                    ya = p * d * numpy.sin(theta) + y1

                a_pos.append(((xa, ya), (x2, y2)))

                #r1 = matplotlib.patches.Rectangle((0, 0), 20, 40, color="blue", alpha=0.50)
                #ax.add_patch(r1)
                # MAYBE DRAW POLYGON INSTEAD OF RECTANGLE

            arrow_collection = LineCollection(a_pos,
                                              colors=arrow_colors,
                                              linewidths=[4 * ww for ww in lw],
                                              antialiaseds=(1,),
                                              transOffset=ax.transData,
                                              )

            arrow_collection.set_zorder(1)  # edges go behind nodes
            arrow_collection.set_label(label)
            ax.add_collection(arrow_collection)

        return edgeCollection, arrow_collection

    '''Modified function networkx.draw_networkx_edges'''

    @staticmethod
    def draw_edges_with_boxes(G, pos,
                   edgelist=None,
                   width=1.0,
                   edge_color='k',
                   style='solid',
                   alpha=1.0,
                   edge_cmap=None,
                   edge_vmin=None,
                   edge_vmax=None,
                   ax=None,
                   arrows=True,
                   label=None,
                   **kwds):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.cbook as cb
            from matplotlib.colors import colorConverter, Colormap
            from matplotlib.collections import LineCollection
            import numpy
        except ImportError:
            raise ImportError("Matplotlib required for draw()")
        except RuntimeError:
            print("Matplotlib unable to open display")
            raise

        if ax is None:
            ax = plt.gca()

        if edgelist is None:
            edgelist = G.edges()

        if not edgelist or len(edgelist) == 0:  # no edges!
            return None

        # set edge positions


        box_pos = numpy.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
        p = 0.25
        edge_pos = []
        for edge in edgelist:
            src, dst = np.array(pos[edge[0]]), np.array(pos[edge[1]])
            s = dst - src
            src = src + p*s
            edge_pos.append((src, dst))
        edge_pos = numpy.asarray(edge_pos)

        if not cb.iterable(width):
            lw = (width,)
        else:
            lw = width

        if not cb.is_string_like(edge_color) \
                and cb.iterable(edge_color) \
                and len(edge_color) == len(edge_pos):
            if numpy.alltrue([cb.is_string_like(c)
                              for c in edge_color]):
                # (should check ALL elements)
                # list of color letters such as ['k','r','k',...]
                edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                     for c in edge_color])
            elif numpy.alltrue([not cb.is_string_like(c)
                                for c in edge_color]):
                # If color specs are given as (rgb) or (rgba) tuples, we're OK
                if numpy.alltrue([cb.iterable(c) and len(c) in (3, 4)
                                  for c in edge_color]):
                    edge_colors = tuple(edge_color)
                else:
                    # numbers (which are going to be mapped with a colormap)
                    edge_colors = None
            else:
                raise ValueError('edge_color must consist of either color names or numbers')
        else:
            if cb.is_string_like(edge_color) or len(edge_color) == 1:
                edge_colors = (colorConverter.to_rgba(edge_color, alpha),)
            else:
                raise ValueError(
                    'edge_color must be a single color or list of exactly m colors where m is the number or edges')
        edge_collection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=width,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        # Note: there was a bug in mpl regarding the handling of alpha values for
        # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
        # r7189 (June 6 2009).  We should then not set the alpha value globally,
        # since the user can instead provide per-edge alphas now.  Only set it
        # globally if provided as a scalar.
        if cb.is_numlike(alpha):
            edge_collection.set_alpha(alpha)

        if edge_colors is None:
            if edge_cmap is not None:
                assert (isinstance(edge_cmap, Colormap))
            edge_collection.set_array(numpy.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
            if edge_vmin is not None or edge_vmax is not None:
                edge_collection.set_clim(edge_vmin, edge_vmax)
            else:
                edge_collection.autoscale()



        if G.is_directed() and arrows:
            box_collection = Utilities.get_boxes(edge_colors=edge_colors, edge_pos=box_pos)
            box_collection.set_zorder(1)  # edges go behind nodes
            box_collection.set_label(label)
            ax.add_collection(box_collection)

        return edge_collection, box_collection

    @staticmethod
    def draw_animation_edges(G, pos,
                   edgelist=None,
                   width=1.0,
                   edge_color='k',
                   style='solid',
                   alpha=1.0,
                   edge_cmap=None,
                   edge_vmin=None,
                   edge_vmax=None,
                   ax=None,
                   arrows=True,
                   label=None,
                   **kwds):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.cbook as cb
            from matplotlib.colors import colorConverter, Colormap
            from matplotlib.collections import LineCollection
            import numpy
        except ImportError:
            raise ImportError("Matplotlib required for draw()")
        except RuntimeError:
            print("Matplotlib unable to open display")
            raise

        if ax is None:
            ax = plt.gca()

        if edgelist is None:
            edgelist = G.edges()

        if not edgelist or len(edgelist) == 0:  # no edges!
            return None

        # set edge positions


        box_pos = numpy.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
        p = 0.25
        edge_pos = []
        for edge in edgelist:
            src, dst = np.array(pos[edge[0]]), np.array(pos[edge[1]])
            s = dst - src
            src = src + p*s
            edge_pos.append((src, dst))
        edge_pos = numpy.asarray(edge_pos)

        if not cb.iterable(width):
            lw = (width,)
        else:
            lw = width

        if not cb.is_string_like(edge_color) \
                and cb.iterable(edge_color) \
                and len(edge_color) == len(edge_pos):
            if numpy.alltrue([cb.is_string_like(c)
                              for c in edge_color]):
                # (should check ALL elements)
                # list of color letters such as ['k','r','k',...]
                edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                     for c in edge_color])
            elif numpy.alltrue([not cb.is_string_like(c)
                                for c in edge_color]):
                # If color specs are given as (rgb) or (rgba) tuples, we're OK
                if numpy.alltrue([cb.iterable(c) and len(c) in (3, 4)
                                  for c in edge_color]):
                    edge_colors = tuple(edge_color)
                else:
                    # numbers (which are going to be mapped with a colormap)
                    edge_colors = None
            else:
                raise ValueError('edge_color must consist of either color names or numbers')
        else:
            if cb.is_string_like(edge_color) or len(edge_color) == 1:
                edge_colors = (colorConverter.to_rgba(edge_color, alpha),)
            else:
                raise ValueError(
                    'edge_color must be a single color or list of exactly m colors where m is the number or edges')
        '''
        modEdgeColors = list(edge_colors)
        modEdgeColors = tuple(modEdgeColors + [colorConverter.to_rgba('w', alpha)
                                     for c in edge_color])
        #print modEdgeColors
        edge_collection = LineCollection(np.asarray(list(edge_pos)*2),
                                         colors=modEdgeColors,
                                         linewidths=[6]*len(list(edge_colors))+[4]*len(list(edge_colors)),
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )
        '''
        edge_collection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=6,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        tube_collection = LineCollection(edge_pos,
                                         colors=tuple([colorConverter.to_rgba('lightgrey', alpha)
                                     for c in edge_color]),
                                         linewidths=4,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         )

        tube_collection.set_zorder(1)  # edges go behind nodes
        tube_collection.set_label(label)
        ax.add_collection(tube_collection)

        # Note: there was a bug in mpl regarding the handling of alpha values for
        # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
        # r7189 (June 6 2009).  We should then not set the alpha value globally,
        # since the user can instead provide per-edge alphas now.  Only set it
        # globally if provided as a scalar.
        if cb.is_numlike(alpha):
            edge_collection.set_alpha(alpha)

        if edge_colors is None:
            if edge_cmap is not None:
                assert (isinstance(edge_cmap, Colormap))
            edge_collection.set_array(numpy.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
            if edge_vmin is not None or edge_vmax is not None:
                edge_collection.set_clim(edge_vmin, edge_vmax)
            else:
                edge_collection.autoscale()



        if G.is_directed() and arrows:
            arrow_collection = Utilities.get_boxes(edge_colors=edge_colors, edge_pos=box_pos)
            arrow_collection.set_zorder(1)  # edges go behind nodes
            arrow_collection.set_label(label)
            ax.add_collection(arrow_collection)

        return edge_collection, arrow_collection, tube_collection

    @staticmethod
    def get_boxes(edge_colors= None, edge_pos=None, width=1.0):
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter
        ax = plt.gca()

        if not cb.iterable(width):
            lw = (width,)
        else:
            lw = width

        if edge_colors is None:
            edge_colors = tuple([colorConverter.to_rgba(c)
                                 for c in 'k'])

        rectangles = []
        arrow_colors = edge_colors
        p = 0.25  # 1/4 of edge should be the box
        radius = 7
        for src, dst in edge_pos:
            src = np.array(src)
            dst = np.array(dst)
            d = np.sqrt(np.sum(((dst - src) * p) ** 2))
            s = dst - src
            if d == 0:  # source and target at same position
                continue
            angle = np.rad2deg(np.arctan2(s[1], s[0]))
            delta = np.array([0, radius])
            t = matplotlib.transforms.Affine2D().rotate_deg_around(src[0], src[1], angle)
            rec = Rectangle(src - delta, width=d, height=radius * 2,
                            transform=t)
            rectangles.append(rec)
            # ax.add_patch(rec)
            # ax.plot([src[0], dst[0]], [src[1], dst[1]], lw=2, solid_capstyle="butt", zorder=0, color='r')

        arrow_collection = PatchCollection(rectangles,
                                           linewidths=[ww for ww in lw],
                                           edgecolors=arrow_colors,
                                           facecolors='none',
                                           antialiaseds=(1,),
                                           transOffset=ax.transData, )
                                            # alpha=0.5)
        return arrow_collection

    @staticmethod
    def draw_nodes(G,
                    pos,
                    nodelist=None,
                    node_size=300,
                    node_color='r',
                    node_shape='o',
                    alpha=1.0,
                    cmap=None,
                    vmin=None,
                    vmax=None,
                    ax=None,
                    linewidths=None,
                    label=None,
                    **kwds):

        try:
            import matplotlib.pyplot as plt
            import numpy
        except ImportError:
            raise ImportError("Matplotlib required for draw()")
        except RuntimeError:
            print("Matplotlib unable to open display")
            raise

        if ax is None:
            ax = plt.gca()

        if nodelist is None:
            nodelist = G.nodes()

        if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
            return None

        try:
            xy = numpy.asarray([pos[v] for v in nodelist])
        except KeyError as e:
            raise nx.NetworkXError('Node %s has no position.'%e)
        except ValueError:
            raise nx.NetworkXError('Bad value in node positions.')

        radius = 7

        circles = []
        for v in nodelist:
            circ = Circle(pos[v], radius=radius, fill=True)
            circles.append(circ)

        node_collection = PatchCollection(circles, facecolors='r', edgecolors='k', linewidths=2, alpha=0.8)
        node_collection.set_zorder(2)
        ax.add_collection(node_collection)
        return node_collection

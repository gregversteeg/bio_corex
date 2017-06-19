""" This module implements some visualizations based on CorEx representations.
"""

import os
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def vis_rep(corex, data=None, row_label=None, column_label=None, prefix='corex_output', focus='', topk=5):
    """Various visualizations and summary statistics for a one layer representation"""
    if column_label is None:
        column_label = map(str, range(data.shape[1]))
    if row_label is None:
        row_label = map(str, range(corex.n_samples))

    alpha = corex.alpha[:, :, 0]

    print 'Groups in sorted_groups.txt'
    output_groups(corex.tcs, alpha, corex.mis, column_label, prefix=prefix)
    output_labels(corex.labels, row_label, prefix=prefix)
    output_cont_labels(corex.p_y_given_x, row_label, prefix=prefix)
    output_strong(corex.tcs, alpha, corex.mis, corex.labels, prefix=prefix)
    anomalies(corex.log_z, row_label=row_label, prefix=prefix)

    if data is not None:
        print 'Pairwise plots among high TC variables in "relationships"'
        data_to_plot = np.where(data == corex.missing_values, np.nan, data)
        cont = cont3(corex.p_y_given_x)
        plot_heatmaps(data_to_plot, corex.labels, alpha, corex.mis, column_label, cont, prefix=prefix, focus=focus)
        plot_pairplots(data_to_plot, corex.labels, alpha, corex.mis, column_label, prefix=prefix, focus=focus, topk=topk)
        plot_top_relationships(data_to_plot, corex.labels, alpha, corex.mis, column_label, cont, prefix=prefix, topk=topk)
        # plot_top_relationships(data_to_plot, corex.labels, alpha, mis, column_label, corex.log_z[:,:,0].T, prefix=prefix+'anomaly_')
    plot_convergence(corex.tc_history, prefix=prefix)


def vis_hierarchy(corexes, row_label=None, column_label=None, max_edges=100, prefix=''):
    """Visualize a hierarchy of representations."""
    if column_label is None:
        column_label = map(str, range(corexes[0].alpha.shape[1]))
    if row_label is None:
        row_label = map(str, range(corexes[0].labels.shape[0]))

    f = safe_open(prefix + '/text_files/higher_layer_group_tcs.txt', 'w+')
    for j, corex in enumerate(corexes):
        f.write('At layer: %d, Total TC: %0.3f\n' % (j, corex.tc))
        f.write('Individual TCS:' + str(corex.tcs) + '\n')
        plot_convergence(corex.tc_history, prefix=prefix, prefix2=j)
        g = safe_open('{}/text_files/mis_layer{}.csv'.format(prefix, j), 'w+')
        h = safe_open('{}/text_files/weights_layer{}.csv'.format(prefix, j), 'w+')
        if j == 0:
            g.write('factor,' + ','.join(column_label) + '\n')
            h.write('factor,' + ','.join(column_label) + '\n')
        else:
            g.write('factor,'+ ','.join(map(str, list(range(len(corex.mis[0,:]))))) + '\n')
            h.write('factor,'+ ','.join(map(str, list(range(len(corex.mis[0,:]))))) + '\n')
        mis = corex.mis / np.log(2)
        alpha = corex.alpha
        for ir, r in enumerate(mis):
            g.write(str(ir) + ',' + ','.join(map(str, mis[ir])) + '\n')
            h.write(str(ir) + ',' + ','.join(map(str, mis[ir] * alpha[ir].ravel())) + '\n')
        g.close()
        h.close()
    f.close()

    import textwrap
    column_label = map(lambda q: '\n'.join(textwrap.wrap(q, width=17, break_long_words=False)), column_label)

    # Construct non-tree graph
    weights = [corex.alpha[:, :, 0].clip(0, 1) * corex.mis for corex in corexes]
    node_weights = [corex.tcs for corex in corexes]
    g = make_graph(weights, node_weights, column_label, max_edges=max_edges)

    # Display pruned version
    h = g.copy()  # trim(g.copy(), max_parents=max_parents, max_children=max_children)
    edge2pdf(h, prefix + '/graphs/graph_prune_' + str(max_edges), labels='label', directed=True, makepdf=True)

    # Display tree version
    tree = g.copy()
    tree = trim(tree, max_parents=1, max_children=False)
    edge2pdf(tree, prefix + '/graphs/tree', labels='label', directed=True, makepdf=True)

    return g


def plot_heatmaps(data, labels, alpha, mis, column_label, cont, topk=20, prefix='', focus=''):
    cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
    m, nv = mis.shape
    for j in range(m):
        inds = np.where(np.logical_and(alpha[j] > 0, mis[j] > 0.))[0]
        inds = inds[np.argsort(- alpha[j, inds] * mis[j, inds])][:topk]
        if focus in column_label:
            ifocus = column_label.index(focus)
            if not ifocus in inds:
                inds = np.insert(inds, 0, ifocus)
        if len(inds) >= 2:
            plt.clf()
            order = np.argsort(cont[:,j])
            subdata = data[:, inds][order].T
            subdata -= np.nanmean(subdata, axis=1, keepdims=True)
            subdata /= np.nanstd(subdata, axis=1, keepdims=True)
            columns = [column_label[i] for i in inds]
            sns.heatmap(subdata, vmin=-3, vmax=3, cmap=cmap, yticklabels=columns, xticklabels=False, mask=np.isnan(subdata))
            filename = '{}/heatmaps/group_num={}.png'.format(prefix, j)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.title("Latent factor {}".format(j))
            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')
            #plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=cont[:, j],
            #          outfile=prefix + '/relationships/group_num=' + str(j), latent=labels[:, j], alpha=0.1)

def plot_pairplots(data, labels, alpha, mis, column_label, topk=5, prefix='', focus=''):
    cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
    plt.rcParams.update({'font.size': 32})
    m, nv = mis.shape
    for j in range(m):
        inds = np.where(np.logical_and(alpha[j] > 0, mis[j] > 0.))[0]
        inds = inds[np.argsort(- alpha[j, inds] * mis[j, inds])][:topk]
        if focus in column_label:
            ifocus = column_label.index(focus)
            if not ifocus in inds:
                inds = np.insert(inds, 0, ifocus)
        if len(inds) >= 2:
            plt.clf()
            subdata = data[:, inds]
            columns = [column_label[i] for i in inds]
            subdata = pd.DataFrame(data=subdata, columns=columns)

            try:
                sns.pairplot(subdata, kind="reg", diag_kind="kde", size=5, dropna=True)
                filename = '{}/pairplots_regress/group_num={}.pdf'.format(prefix, j)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                plt.suptitle("Latent factor {}".format(j), y=1.01)
                plt.savefig(filename, bbox_inches='tight')
                plt.clf()
            except:
                pass

            subdata['Latent factor'] = labels[:,j]
            try:
                sns.pairplot(subdata, kind="scatter", dropna=True, vars=subdata.columns.drop('Latent factor'), hue="Latent factor", diag_kind="kde", size=5)
                filename = '{}/pairplots/group_num={}.pdf'.format(prefix, j)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                plt.suptitle("Latent factor {}".format(j), y=1.01)
                plt.savefig(filename, bbox_inches='tight')
                plt.close('all')
            except:
                pass



def make_graph(weights, node_weights, column_label, max_edges=100):
    all_edges = np.hstack(map(np.ravel, weights))
    max_edges = min(max_edges, len(all_edges))
    w_thresh = np.sort(all_edges)[-max_edges]
    print 'weight threshold is %f for graph with max of %f edges ' % (w_thresh, max_edges)
    g = nx.DiGraph()
    max_node_weight = max([max(w) for w in node_weights])
    for layer, weight in enumerate(weights):
        m, n = weight.shape
        for j in range(m):
            g.add_node((layer + 1, j))
            g.node[(layer + 1, j)]['weight'] = 0.3 * node_weights[layer][j] / max_node_weight
            for i in range(n):
                if weight[j, i] > w_thresh:
                    if weight[j, i] > w_thresh / 2:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 10 * weight[j, i])])
                    else:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 0)])

    # Label layer 0
    for i, lab in enumerate(column_label):
        g.add_node((0, i))
        g.node[(0, i)]['label'] = lab
        g.node[(0, i)]['name'] = lab  # JSON uses this field
        g.node[(0, i)]['weight'] = 1
    return g


def trim(g, max_parents=False, max_children=False):
    for node in g:
        if max_parents:
            parents = list(g.successors(node))
            weights = [g.edge[node][parent]['weight'] for parent in parents]
            for weak_parent in np.argsort(weights)[:-max_parents]:
                g.remove_edge(node, parents[weak_parent])
        if max_children:
            children = g.predecessors(node)
            weights = [g.edge[child][node]['weight'] for child in children]
            for weak_child in np.argsort(weights)[:-max_children]:
                g.remove_edge(children[weak_child], node)
    return g


def output_groups(tcs, alpha, mis, column_label, thresh=0, prefix=''):
    f = safe_open(prefix + '/text_files/groups.txt', 'w+')
    g = safe_open(prefix + '/text_files/groups_no_overlaps.txt', 'w+')
    m, nv = mis.shape
    for j in range(m):
        f.write('Group num: %d, TC(X;Y_j): %0.3f\n' % (j, tcs[j]))
        g.write('Group num: %d, TC(X;Y_j): %0.3f\n' % (j, tcs[j]))
        inds = np.where(alpha[j] * mis[j] > thresh)[0]
        inds = inds[np.argsort(-alpha[j, inds] * mis[j, inds])]
        for ind in inds:
            f.write(column_label[ind] + ', %0.3f, %0.3f, %0.3f\n' % (
                mis[j, ind], alpha[j, ind], mis[j, ind] * alpha[j, ind]))
        inds = np.where(alpha[j] == 1)[0]
        inds = inds[np.argsort(- mis[j, inds])]
        for ind in inds:
            g.write(column_label[ind] + ', %0.3f\n' % mis[j, ind])
    f.close()
    g.close()


def output_labels(labels, row_label, prefix=''):
    f = safe_open(prefix + '/text_files/labels.txt', 'w+')
    ns, m = labels.shape
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(map(str, labels[l, :])) + '\n')
    f.close()


def output_cont_labels(p_y_given_x, row_label, prefix=''):
    f = safe_open(prefix + '/text_files/cont_labels.txt', 'w+')
    m, ns, k = p_y_given_x.shape
    # assert k==2, 'More complicated if k>2... use cont3_test to generate if k==3'
    labels = cont3(p_y_given_x)
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(map(lambda q: '%0.6f' % q, labels[l, :])) + '\n')
    f.close()


def output_strong(tcs, alpha, mis, labels, prefix=''):
    f = safe_open(prefix + '/text_files/most_deterministic_groups.txt', 'w+')
    m, n = alpha.shape
    topk = 5
    ixy = np.clip(np.sum(alpha * mis, axis=1) - tcs, 0, np.inf)
    hys = np.array([entropy(labels[:, j]) for j in range(m)]).clip(1e-6)
    ntcs = [(np.sum(np.sort(alpha[j] * mis[j])[-topk:]) - ixy[j]) / ((topk - 1) * hys[j]) for j in range(m)]

    f.write('Group num., NTC\n')
    for j, ntc in sorted(enumerate(ntcs), key=lambda q: -q[1]):
        f.write('%d, %0.3f\n' % (j, ntc))
    f.close()


def plot_top_relationships(data, labels, alpha, mis, column_label, cont, topk=5, athresh=0.2, prefix=''):
    m, nv = mis.shape
    for j in range(m):
        # inds = np.where(alpha[j] * mis[j] > 0)[0]
        #inds = inds[np.argsort(-alpha[j, inds] * mis[j, inds])][:topk]
        inds = np.where(alpha[j] > athresh)[0]
        inds = inds[np.argsort(- alpha[j, inds] * mis[j, inds])][:topk]
        if len(inds) >= 2:
            plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=cont[:, j],
                      outfile=prefix + '/relationships/group_num=' + str(j), latent=labels[:, j], alpha=0.1)


def anomalies(log_z, row_label=None, prefix=''):
    from scipy.special import erf

    ns = log_z.shape[1]
    if row_label is None:
        row_label = map(str, range(ns))
    a_score = np.sum(log_z[:, :, 0], axis=0)
    mean, std = np.mean(a_score), np.std(a_score)
    a_score = (a_score - mean) / std
    percentile = 1. / ns
    anomalies = np.where(0.5 * (1 - erf(a_score / np.sqrt(2)) ) < percentile)[0]
    f = safe_open(prefix + '/text_files/anomalies.txt', 'w+')
    for i in anomalies:
        f.write(row_label[i] + ', %0.1f\n' % a_score[i])
    f.close()


def compact(data, fraction=0.5, missing_values=-1e6):
    # Return indices and data matrix only for rows with at least 50% of the columns non-empty
    ns, nv = data.shape
    inds = np.sum(data == missing_values, axis=1) < fraction * nv
    return inds, data[inds]


# Utilities
# IT UTILITIES
def entropy(xsamples):
    # sample entropy for one discrete var
    xsamples = np.asarray(xsamples)
    xsamples = xsamples[xsamples >= 0]  # by def, -1 means missing value
    xs = np.unique(xsamples)
    ns = len(xsamples)
    ps = np.array([float(np.count_nonzero(xsamples == x)) / ns for x in xs])
    return -np.sum(ps * np.log(ps))


def safe_open(filename, mode):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return open(filename, mode)


# Visualization utilities

def neato(fname, position=None, directed=False):
    if directed:
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=False -o " + fname + "_sfdp.pdf")
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=True -o " + fname + "_sfdp_w_splines.pdf")
        return True
    if position is None:
        os.system("neato " + fname + ".dot -Tpdf -o " + fname + ".pdf")
        os.system("fdp " + fname + ".dot -Tpdf -o " + fname + "fdp.pdf")
    else:
        os.system("neato " + fname + ".dot -Tpdf -n -o " + fname + ".pdf")
    return True


def extract_color(label):
    colors = 'indigo,gold,hotpink,firebrick,indianred,yellow,mistyrose,darkolivegreen,darkseagreen,pink,tomato,lightcoral,orangered,navajowhite,palegreen,darkslategrey,greenyellow,burlywood,seashell,mediumspringgreen,papayawhip,blanchedalmond,chartreuse,dimgray,black,peachpuff,springgreen,aquamarine,white,orange,lightsalmon,darkslategray,brown,ivory,dodgerblue,peru,lawngreen,chocolate,crimson,forestgreen,slateblue,lightseagreen,cyan,mintcream,antiquewhite,mediumorchid,skyblue,gray,darkturquoise,goldenrod,darkgreen,floralwhite,darkviolet,moccasin,saddlebrown,grey,darkslateblue,lightskyblue,lightpink,mediumvioletred,slategrey,red,deeppink,limegreen,palegoldenrod,plum,turquoise,lightgrey,lightgoldenrodyellow,darkgoldenrod,lavender,maroon,yellowgreen,sandybrown,thistle,violet,navy,magenta,dimgrey,tan,rosybrown,blue,lightblue,ghostwhite,honeydew,cornflowerblue,linen,powderblue,seagreen,darkkhaki,snow,sienna,mediumblue,royalblue,lightcyan,green,mediumpurple,midnightblue,cornsilk,paleturquoise,bisque,slategray,khaki,wheat,darkorchid,deepskyblue,salmon,steelblue,palevioletred,lightslategray,aliceblue,lightslategrey,orchid,gainsboro,mediumseagreen,lightgray,mediumturquoise,lemonchiffon,cadetblue,lightyellow,lavenderblush,coral,purple,whitesmoke,mediumslateblue,darkorange,mediumaquamarine,darksalmon,beige,blueviolet,azure,lightsteelblue,oldlace'.split(',')

    parts = label.split('_')
    for part in parts:
        if part in colors:
            parts.remove(part)
            return '_'.join(parts), part
    return label, 'black'


def edge2pdf(g, filename, threshold=0, position=None, labels=None, connected=True, directed=False, makepdf=True):
    #This function will takes list of edges and a filename
    #and write a file in .dot format. Readable, eg. by omnigraffle
    # OR use "neato file.dot -Tpng -n -o file.png"
    # The -n option says whether to use included node positions or to generate new ones
    # for a grid, positions = [(i%28,i/28) for i in range(784)]
    def cnn(node):
        #change node names for dot format
        if type(node) is tuple or type(node) is list:
            return u'n' + u'_'.join(map(unicode, node))
        else:
            return unicode(node)

    if connected:
        touching = list(set(sum([[a, b] for a, b in g.edges()], [])))
        g = nx.subgraph(g, touching)
        print 'non-isolated nodes,edges', len(list(g.nodes())), len(list(g.edges()))
    f = safe_open(filename + '.dot', 'w+')
    if directed:
        f.write("strict digraph {\n".encode('utf-8'))
    else:
        f.write("strict graph {\n".encode('utf-8'))
    #f.write("\tgraph [overlap=scale];\n".encode('utf-8'))
    f.write("\tnode [shape=point];\n".encode('utf-8'))
    for a, b, d in g.edges(data=True):
        if d.has_key('weight'):
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ' [penwidth=%.2f' % float(
                    np.clip(d['weight'], 0, 9)) + '];\n').encode('utf-8'))
            else:
                if d['weight'] > threshold:
                    f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ' [penwidth=' + str(3 * d['weight']) + '];\n').encode(
                        'utf-8'))
        else:
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ';\n').encode('utf-8'))
            else:
                f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ';\n').encode('utf-8'))
    for n in g.nodes():
        if labels is not None:
            if type(labels) == dict or type(labels) == list:
                thislabel = labels[n].replace(u'"', u'\\"')
                lstring = u'label="' + thislabel + u'",shape=none'
            elif type(labels) == str:
                if g.node[n].has_key('label'):
                    thislabel = g.node[n][labels].replace(u'"', u'\\"')
                    # combine dupes
                    #llist = thislabel.split(',')
                    #thislabel = ','.join([l for l in set(llist)])
                    thislabel, thiscolor = extract_color(thislabel)
                    lstring = u'label="%s",shape=none,fontcolor="%s"' % (thislabel, thiscolor)
                else:
                    weight = g.node[n].get('weight', 0.1)
                    if n[0] == 1:
                        lstring = u'shape=circle,margin="0,0",style=filled,fillcolor=black,fontcolor=white,height=%0.2f,label="%d"' % (
                            2 * weight, n[1])
                    else:
                        lstring = u'shape=point,height=%0.2f' % weight
            else:
                lstring = 'label="' + str(n) + '",shape=none'
            lstring = unicode(lstring)
        else:
            lstring = False
        if position is not None:
            if position == 'grid':
                position = [(i % 28, 28 - i / 28) for i in range(784)]
            posstring = unicode('pos="' + str(position[n][0]) + ',' + str(position[n][1]) + '"')
        else:
            posstring = False
        finalstring = u' [' + u','.join([ts for ts in [posstring, lstring] if ts]) + u']\n'
        #finalstring = u' ['+lstring+u']\n'
        f.write((u'\t' + cnn(n) + finalstring).encode('utf-8'))
    f.write("}".encode('utf-8'))
    f.close()
    if makepdf:
        neato(filename, position=position, directed=directed)
    return True


def predictable(out, data, wdict=None, topk=5, outfile='sorted_groups.txt', graphs=False, prefix='', athresh=0.5,
                tvalue=0.1):
    alpha, labels, lpygx, mis, lasttc = out[:5]
    ns, m = labels.shape
    m, nv = mis.shape
    hys = [entropy(labels[:, j]) for j in range(m)]
    #alpha = np.array([z[2] for z in zs]) # m by nv
    nmis = []
    ixys = []
    for j in range(m):
        if hys[j] > 0:
            #ixy = np.dot((alpha[j]>0.95).astype(int),mis[j])-lasttc[-1][j]
            ixy = max(0., np.dot(alpha[j], mis[j]) - lasttc[-1][j])
            ixys.append(ixy)
            tcn = (np.sum(np.sort(alpha[j] * mis[j])[-topk:]) - ixy) / ((topk - 1) * hys[j])
            nmis.append(tcn)  #ixy) #/hys[j])
        else:
            ixys.append(0)
            nmis.append(0)
    f = safe_open(prefix + outfile, 'w+')
    print list(enumerate(np.argsort(-np.array(nmis))))
    print ','.join(map(str, list(np.argsort(-np.array(nmis)))))
    for i, top in enumerate(np.argsort(-np.array(nmis))):
        f.write('Group num: %d, Score: %0.3f\n' % (top, nmis[top]))
        inds = np.where(alpha[top] > athresh)[0]
        inds = inds[np.argsort(-mis[top, inds])]
        for ind in inds:
            f.write(wdict[ind] + ', %0.3f\n' % (mis[top, ind] / np.log(2)))
        if wdict:
            print ','.join(map(lambda q: wdict[q], inds))
            print ','.join(map(str, inds))
        print top
        print nmis[top], ixys[top], hys[top], ixys[top] / hys[top]  #,lasttc[-1][top],hys[top],lasttc[-1][top]/hys[top]
        if graphs:
            print inds
            if len(inds) >= 2:
                plot_rels(data[:, inds[:5]], map(lambda q: wdict[q], inds[:5]),
                          outfile='relationships/' + str(i) + '_group_num=' + str(top), latent=out[1][:, top],
                          alpha=tvalue)
    f.close()
    return nmis


def shorten(s, n=12):
    s, _ = extract_color(s)
    if len(s) > 2 * n:
        return s[:n] + '..' + s[-n:]
    return s


def plot_convergence(tc_history, prefix='', prefix2=''):
    pylab.plot(tc_history)
    pylab.xlabel('# iterations')
    filename = '{}/text_files/convergence{}.pdf'.format(prefix, prefix2)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pylab.savefig(filename)
    pylab.close('all')
    return True


def plot_rels(data, labels=None, colors=None, outfile="rels", latent=None, alpha=0.8):
    ns, n = data.shape
    if labels is None:
        labels = map(str, range(n))
    ncol = 5
    # ncol = 4
    nrow = int(np.ceil(float(n * (n - 1) / 2) / ncol))
    #nrow=1
    #pylab.rcParams.update({'figure.autolayout': True})
    fig, axs = pylab.subplots(nrow, ncol)
    fig.set_size_inches(5 * ncol, 5 * nrow)
    #fig.set_canvas(pylab.gcf().canvas)
    pairs = list(combinations(range(n), 2))  #[:4]
    pairs = sorted(pairs, key=lambda q: q[0]**2+q[1]**2)  # Puts stronger relationships first
    if colors is not None:
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors)).clip(1e-7)

    for ax, pair in zip(axs.flat, pairs):
        if latent is None:
            ax.scatter(data[:, pair[0]], data[:, pair[1]], marker='.', edgecolors='none', alpha=alpha)
        else:
            # cs = 'rgbcmykrgbcmyk'
            markers = 'x+.o,<>^^<>,+x.'
            for j, ind in enumerate(np.unique(latent)):
                inds = (latent == ind)
                ax.scatter(data[inds, pair[0]], data[inds, pair[1]], c=colors[inds], cmap=pylab.get_cmap("jet"),
                           marker=markers[j], alpha=0.5, edgecolors='none', vmin=0, vmax=1)

        ax.set_xlabel(shorten(labels[pair[0]]))
        ax.set_ylabel(shorten(labels[pair[1]]))

    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.scatter(data[:, 0], data[:, 1], marker='.')

    pylab.rcParams['font.size'] = 12  #6
    pylab.draw()
    #fig.set_tight_layout(True)
    fig.tight_layout()
    for ax in axs.flat[axs.size - 1:len(pairs) - 1:-1]:
        ax.set_visible(False)
    filename = outfile + '.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    fig.savefig(outfile + '.png')  #df')
    pylab.close('all')
    return True


def cont3(p_y_given_x):
    """
    Returns an ordering for points in a simplex (using manifold learning methods).
    This is used for shading points in relationships/ plots.
    """
    nv, ns, k = p_y_given_x.shape
    allcont = []
    if k == 3:
        for j in range(nv):
            ends = sorted([(np.dot(p_y_given_x[j, :, end[0]], p_y_given_x[j, :, end[1]]), end) for end in
                           [(0, 1), (0, 2), (1, 2)]])[0][1]
            cont = np.log(p_y_given_x[j, :, ends[0]]) - np.log(p_y_given_x[j, :, ends[1]])
            cont = np.where(np.isnan(cont), 0, cont)
            allcont.append(cont)
        out = np.array(allcont).T  # Bounds approximately reflect log of float precision
        top = np.max(np.abs(np.where(np.isinf(out), 0, out)))
        out = np.clip(out, -top, top)
    elif k == 2:
        for j in range(nv):
            Y = p_y_given_x[j, :, 0]  #np.log(p_y_given_x[j,:,0].clip(1e-40))
            allcont.append(Y)
        out = np.array(allcont).T
    else:
        print 'error, not able to visualize with k > 3'
    return out

def cont3_test(p_y_given_x, p_test):
    """
    What if we need the same continuous representation for test data? If k==3,(or above) we have to make sure we
    do the same transformation as the reference data.
    """
    nv, ns, k = p_y_given_x.shape
    allcont = []
    assert k==3
    for j in range(nv):
        ends = sorted([(np.dot(p_y_given_x[j, :, end[0]], p_y_given_x[j, :, end[1]]), end) for end in
                       [(0, 1), (0, 2), (1, 2)]])[0][1]
        cont = np.log(p_test[j, :, ends[0]]) - np.log(p_test[j, :, ends[1]])
        cont = np.where(np.isnan(cont), 0, cont)
        allcont.append(cont)
    out = np.array(allcont).T  # Bounds approximately reflect log of float precision
    top = np.max(np.abs(np.where(np.isinf(out), 0, out)))
    out = np.clip(out, -top, top)
    return out

if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python vis_corex.py data/test_data.csv
    # python vis_corex.py tests/data/test_big5.csv --layers=5,1 -v --no_row_names -o big5
    import corex as ce
    import csv
    import sys, traceback
    from time import time
    from optparse import OptionParser, OptionGroup

    parser = OptionParser(usage="usage: %prog [options] data_file.csv \n"
                                "It is assumed that the first row and first column of the data CSV file are labels.\n"
                                "Use options to indicate otherwise.")
    group = OptionGroup(parser, "Input Data Format Options")
    group.add_option("-c", "--continuous",
                     action="store_true", dest="continuous", default=False,
                     help="Input variables are continuous (default assumption is that they are discrete).")
    group.add_option("-t", "--no_column_names",
                     action="store_true", dest="nc", default=False,
                     help="We assume the top row is variable names for each column. "
                          "This flag says that data starts on the first row and gives a "
                          "default numbering scheme to the variables (1,2,3...).")
    group.add_option("-f", "--no_row_names",
                     action="store_true", dest="nr", default=False,
                     help="We assume the first column is a label or index for each sample. "
                          "This flag says that data starts on the first column.")
    group.add_option("-m", "--missing",
                     action="store", dest="missing", type="float", default=-1e6,
                     help="Treat this value as missing data. Default is -1e6.")
    group.add_option("-d", "--delimiter",
                     action="store", dest="delimiter", type="string", default=",",
                     help="Separator between entries in the data, default is ','.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "CorEx Options")
    group.add_option("-l", "--layers", dest="layers", type="string", default="2,1",
                     help="Specify number of units at each layer: 5,3,1 has "
                          "5 units at layer 1, 3 at layer 2, and 1 at layer 3")
    group.add_option("-k", "--dim_hidden", dest="dim_hidden", type="int", default=2,
                     help="Latent factors take values 0, 1..k. Default k=2")

    group.add_option("-b", "--bayesian_smoothing",
                     action="store_true", dest="smooth", default=False,
                     help="Turn on Bayesian smoothing when estimating marginal distributions (p(x_i|y_j)). "
                          "Slower, but reduces appearance of spurious correlations if the number of "
                          "samples is < 200 or if dim_hidden is large.")
    group.add_option("-r", "--repeat",
                     action="store", dest="repeat", type="int", default=1,
                     help="Run r times and return solution with best TC.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="corex_output",
                     help="A directory to put all output files.")
    group.add_option("-v", "--verbose",
                     action="store_true", dest="verbose", default=False,
                     help="Print rich outputs while running.")
    group.add_option("-e", "--edges",
                     action="store", dest="max_edges", type="int", default=100,
                     help="Show at most this many edges in graphs.")
    group.add_option("-q", "--regraph",
                     action="store_true", dest="regraph", default=False,
                     help="Don't re-run corex, just re-generate outputs (perhaps with edges option changed).")
    group.add_option("-F", "--focus",
                     action="store", dest="focus", type="string", default="",
                     help="A special variable to focus on in plots.")
    group.add_option("-T", "--topk",
                     action="store", dest="topk", type=int, default=5,
                     help="How many variables to look at in pairplots.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Computational Options")
    group.add_option("-a", "--ram",
                     action="store", dest="ram", type="float", default=8.,
                     help="Approximate amount of RAM to use (in GB).")
    group.add_option("-p", "--cpu",
                     action="store", dest="cpu", type="int", default=1,
                     help="Number of cpus/cores to use.")
    group.add_option("-w", "--max_iter",
                     action="store", dest="max_iter", type="int", default=100,
                     help="Max number of iterations to use.")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if not len(args) == 1:
        print "Run with '-h' option for usage help."
        sys.exit()

    np.set_printoptions(precision=3, suppress=True)  # For legible output from numpy
    layers = map(int, options.layers.split(','))
    if layers[-1] != 1:
        layers.append(1)  # Last layer has one unit for convenience so that graph is fully connected.
    verbose = options.verbose

    def fill_empty(z):
        if z=='':
            return str(options.missing)
        else:
            return z

    #Load data from csv file
    filename = args[0]
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=options.delimiter)
        if options.nc:
            variable_names = None
        else:
            variable_names = reader.next()[(1 - options.nr):]
        sample_names = []
        data = []
        for row in reader:
            if options.nr:
                sample_names = None
            else:
                sample_names.append(row[0])
            tmp = map(fill_empty, row[(1 - options.nr):])
            data.append(tmp)

    try:
        if options.continuous:
            X = np.array(data, dtype=float)  # Data matrix in numpy format
            marg = 'gaussian'
        else:
            X = np.array(data, dtype=int)  # Data matrix in numpy format
            marg = 'discrete'
    except:
        print "Incorrect data format.\nCheck that you've correctly specified options " \
              "such as continuous or not, \nand if there is a header row or column.\n" \
              "Also, missing values should be specified with a numeric value (-1 by default).\n" \
              "Run 'python vis_corex.py -h' option for help with options."
        traceback.print_exc(file=sys.stdout)
        sys.exit()

    if verbose:
        print '\nData summary: X has %d rows and %d columns' % X.shape
        if variable_names:
            print 'Variable names are: ' + ','.join(map(str, list(enumerate(variable_names))))

    # Run CorEx on data
    if verbose:
        print 'Getting CorEx results'
        corexes = []
    if not options.regraph:
        for l, layer in enumerate(layers):
            if verbose:
                print "Layer ", l
            if l == 0:
                t0 = time()
                corexes = [ce.Corex(n_hidden=layer, dim_hidden=options.dim_hidden,
                                    verbose=verbose, marginal_description=marg,
                                    smooth_marginals=options.smooth,
                                    missing_values=options.missing, n_repeat=options.repeat, max_iter=options.max_iter,
                                    n_cpu=options.cpu, ram=options.ram).fit(X)]
                print 'Time for first layer: %0.2f' % (time() - t0)
            else:
                X_prev = corexes[-1].labels
                corexes.append(ce.Corex(n_hidden=layer, dim_hidden=options.dim_hidden,
                                        verbose=verbose, marginal_description='discrete',
                                        smooth_marginals=options.smooth,
                                        n_repeat=options.repeat,
                                        n_cpu=options.cpu, ram=options.ram).fit(X_prev))
        for l, corex in enumerate(corexes):
            # The learned model can be loaded again using ce.Corex().load(filename)
            print 'TC at layer %d is: %0.3f' % (l, corex.tc)
            corex.save(options.output + '/layer_' + str(l) + '.dat')
    else:
        corexes = [ce.Corex().load(options.output + '/layer_' + str(l) + '.dat') for l in range(len(layers))]

    # This line outputs plots showing relationships at the first layer
    vis_rep(corexes[0], X, row_label=sample_names, column_label=variable_names, prefix=options.output, focus=options.focus, topk=options.topk)
    # This line outputs a hierarchical networks structure in a .dot file in the "graphs" folder
    # And it tries to compile the dot file into a pdf using the command line utility sfdp (part of graphviz)
    vis_hierarchy(corexes, row_label=sample_names, column_label=variable_names, max_edges=options.max_edges,
                  prefix=options.output)

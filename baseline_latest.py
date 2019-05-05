import networkx as nx
from linkpred.predictors.eigenvector import RootedPageRank, SimRank
from linkpred.predictors import CommonNeighbours, AdamicAdar, Pearson, Jaccard, DegreeProduct
from networkx.algorithms import preferential_attachment
from linkpred.predictors.path import GraphDistance, Katz
import matplotlib.pyplot as plt
from linkpred.predictors import Random
import pickle
import random
import parameters as params
import networkit as nkt
from networkit.nxadapter import nx2nk

num_links_to_predict = 10000


def build_network():
    G = nx.read_gpickle('/home/social-sim/Info_Diff/data_latest/graph_2016-08-01_2017-01-01.pkl')
    node_groups = pickle.load(open('/home/social-sim/Info_Diff/data_latest/node_groups_2016-08-01_2017-01-01.pkl', 'rb'))
    users = node_groups['users']
    edges_to_remove = []
    for edge in G.edges():
        if edge[0] in users and edge[1] in users:
            edges_to_remove.append((edge[0], edge[1]))
    G.remove_edges_from(edges_to_remove)
    node_mapping = {}
    for i, node in enumerate(list(G)):
        node_mapping[node] = i

    nkt_graph = nkt.graph.Graph()
    for edge in G.edges():
        print(node_mapping[edge[0]], node_mapping[edge[1]])
        nkt_graph.addEdge(node_mapping[edge[0]], node_mapping[edge[1]])
    return node_mapping, nkt_graph


def predict_links_simrank(G):
    simrank = SimRank(G, excluded=G.edges())
    simrank_results = simrank.predict(c=0.4)

    top_k = simrank_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_graph_distance(G):
    graph_distance = GraphDistance(G)
    gd_results = graph_distance.predict(weight=None)
    top_k = gd_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_rooted_pagerank(G):
    rooted_pagerank = RootedPageRank(G)
    simrank_results = rooted_pagerank.predict()
    top_k = simrank_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_common_neighbourhood(G):
    common_neighbours = CommonNeighbours(G)
    simrank_results = common_neighbours.predict()
    top_k = simrank_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_katz(G):
    katz = Katz(G)
    katz_results = katz.predict()
    top_k = katz_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_common_neighbours(G):
    common_neighbours = CommonNeighbours(G)
    simrank_results = common_neighbours.predict()
    top_k = simrank_results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_adamic_adar(G):
    adamic_adar = AdamicAdar(G)
    results = adamic_adar.predict()
    top_k = results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def get_key(item):
    return item[2]


def predict_preferential_attachment(G):
    dp = DegreeProduct(G)
    results = dp.predict()
    top_k = results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links

    #results = preferential_attachment(G)
    #results = sorted(results, key=get_key)
    #predictions = results[0:num_links_to_predict]
    #return predictions


def predict_links_jaccard(G):
    jaccard = Jaccard(G)
    results = jaccard.predict()
    top_k = results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_pearson(G):
    pearson = Pearson(G)
    results = pearson.predict()
    top_k = results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def predict_links_random(G):
    random = Random(G)
    results = random.predict()
    top_k = results.top(num_links_to_predict)
    links = []
    for link, score in top_k.items():
        links.append(link)
    return links


def test_prediction(samples_positive, samples_negative, links):
    hits = 0
    random.shuffle(samples_positive)
    for sample in samples_positive[0:params.count]:
        repo = sample[0]
        user = sample[2]
        for link in links:
            if repo in link and user in link:
                hits += 1
    acc_positive = float(hits)/len(samples_positive)
    #hits = 0
    #random.shuffle(samples_negative)
    #for sample in samples_negative[0:params.count]:
    #    repo = sample[0]
    #    user = sample[2]
    #    present = False
    #    for link in links:
    #        if repo in link and user in link:
    #            present = True
    #    if not present:
    #        hits += 1
    #acc_negative = float(hits)/len(samples_negative)
    #print acc_positive, acc_negative
    #return float(acc_positive + acc_negative)/2
    return acc_positive


def test_baseline():
    G = build_network()
    samples_positive = pickle.load(open('/home/social-sim/Info_Diff/data_latest/test_data_positive_samples.pkl'))
    samples_negative = pickle.load(open('/home/social-sim/Info_Diff/data_latest/test_data_negative_samples.pkl'))

    links = predict_links_jaccard(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, jaccard", result)

    links = predict_links_simrank(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, simrank", result)

    links = predict_links_common_neighbourhood(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, common_neighbourhood", result)

    links = predict_links_rooted_pagerank(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, rooted_pagerank", result)

    links = predict_links_katz(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, katz", result)

    links = predict_links_graph_distance(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, graph_distance", result)

    links = predict_links_adamic_adar(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, adamic_adar", result)

    links = predict_links_pearson(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, pearson", result)

    links = predict_links_random(G)
    result = test_prediction(samples_positive, samples_negative, links)
    print("result, random", result)


def visualize():
    G = build_network()
    nx.draw_networkx(G, node_size=10, with_labels=False, pos=nx.spring_layout(G, k=0.05))
    plt.show()


if __name__ == '__main__':
    #test_baseline()
    node_mapping, g = build_network()
    print(len(g.nodes))
    samples_positive = pickle.load(open('/home/social-sim/Info_Diff/data_latest/test_data_positive_samples.pkl', 'rb'))
    samples_negative = pickle.load(open('/home/social-sim/Info_Diff/data_latest/test_data_negative_samples.pkl', 'rb'))
    for edge in g.edges():
        predictions = nkt.linkprediction.JaccardIndex(g).run(edge[0], edge[1])
        if predictions != 0:
            print(edge)
            print(predictions)

    #links = predict_preferential_attachment(G)
    #result = test_prediction(samples_positive, samples_negative, links)
    #print("result, preferentail attachment", result)